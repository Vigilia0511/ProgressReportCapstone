using Plugin.LocalNotification;
using System;
using System.Collections.ObjectModel;
using Microsoft.Maui.Controls;
using MySql.Data.MySqlClient;
using System.Timers;
using Microsoft.Maui.ApplicationModel; // For MainThread
using Microsoft.Maui.Storage; // For Preferences
using testdatabase.Helpers;  // <-- Add this to access ThemeManager
using Plugin.Maui.Audio; // Add this NuGet package for audio recording
using SocketIOClient;    // Add this NuGet package for Socket.IO

namespace testdatabase
{
    public partial class DPage : ContentPage
    {
        private bool _isSidebarVisible = false;
        private System.Timers.Timer _refreshTimer;
        private System.Timers.Timer _approvalCheckTimer; // New timer for checking approval status
        private ObservableCollection<string> _notifications = new();
        private readonly string _connStr = "server=10.83.177.213;user=root;password=oneinamillion;database=smartdb;";
        private Random _random = new Random();
        private string _loggedInUser = string.Empty;
        private bool isSettingsExpanded = false;
        private bool _isLoadingSettings = false; // Add this flag to prevent event triggers during loading
        private IAudioRecorder _audioRecorder;
        private MemoryStream _audioStream;
        private SocketIOClient.SocketIO _socket;
        private bool _isListeningToIntercom = false;

        private Button IntercomListenButton;

        // ADDING INTERCOM STREAMING VARIABLES
        private bool _isIntercomStreaming = false;
        private CancellationTokenSource _intercomCts;

        public DPage()
        {
            InitializeComponent();

            IntercomListenButton = this.FindByName<Button>("IntercomListenButton");

            LoadSettings();

            Shell.SetBackButtonBehavior(this, new BackButtonBehavior { IsVisible = false, IsEnabled = false });

            // Get logged in user
            _loggedInUser = Preferences.Get("LoggedInUser", string.Empty);

            // Update sidebar with username
            UpdateSidebarUsername();

            // Initial load of notifications
            LoadNotificationsAsync(); // Call async version on start

            // Set up refresh timer for notifications
            _refreshTimer = new System.Timers.Timer(10000); // 10 seconds
            _refreshTimer.Elapsed += RefreshTimer_Elapsed;
            _refreshTimer.AutoReset = true;
            _refreshTimer.Start();

            // Set up approval check timer
            _approvalCheckTimer = new System.Timers.Timer(15000); // 15 seconds
            _approvalCheckTimer.Elapsed += ApprovalCheckTimer_Elapsed;
            _approvalCheckTimer.AutoReset = true;
            _approvalCheckTimer.Start();

            // Initialize audio recorder
            _audioRecorder = AudioManager.Current.CreateRecorder();

            // Initialize socket connection to your Pi
            _socket = new SocketIOClient.SocketIO("http://192.168.137.135:5000");
            _ = _socket.ConnectAsync();

            _socket.On("intercom_stream", async response =>
            {
                var data = response.GetValue<System.Text.Json.JsonElement>();
                var audioBase64 = data.GetProperty("audio").GetString();
                var sampleRate = data.GetProperty("sample_rate").GetInt32();
                var audioBytes = Convert.FromBase64String(audioBase64);
                await PlayAudioBytes(audioBytes, sampleRate);
            });

            // Add this event handler after initializing the socket in your constructor:
            _socket.OnDisconnected += (sender, args) =>
            {
                MainThread.BeginInvokeOnMainThread(async () =>
                {
                    await DisplayAlert("Intercom Error", "Connection to the server was lost. Please check your network or server status.", "OK");
                });
            };

            // Optionally, add automatic reconnect logic:
            _socket.OnDisconnected += async (sender, args) =>
            {
                int retryCount = 0;
                while (!_socket.Connected && retryCount < 5)
                {
                    try
                    {
                        await Task.Delay(2000);
                        await _socket.ConnectAsync();
                        retryCount++;
                    }
                    catch
                    {
                        // Ignore and retry
                    }
                }
                if (!_socket.Connected)
                {
                    MainThread.BeginInvokeOnMainThread(async () =>
                    {
                        await DisplayAlert("Intercom Error", "Unable to reconnect to the server after several attempts.", "OK");
                    });
                }
            };
        }

        private void UpdateSidebarUsername()
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(_loggedInUser))
                {
                    // Update the sidebar username label
                    if (SidebarUsernameLabel != null)
                    {
                        SidebarUsernameLabel.Text = $" {_loggedInUser}";
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating sidebar username: {ex.Message}");
            }
        }

        private void RefreshTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            MainThread.BeginInvokeOnMainThread(async () =>
            {
                await LoadNotificationsAsync();
            }); 
        }

        private void ApprovalCheckTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            MainThread.BeginInvokeOnMainThread(async () =>
            {
                await CheckUserApprovalStatus();
            });
        }

        private async Task CheckUserApprovalStatus()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(_loggedInUser))
                    return;

                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                string query = "SELECT is_approved FROM users WHERE LOWER(username) = LOWER(@username)";
                using var cmd = new MySqlCommand(query, conn);
                cmd.Parameters.AddWithValue("@username", _loggedInUser);

                var result = await cmd.ExecuteScalarAsync();

                if (result != null)
                {
                    bool isApproved = Convert.ToBoolean(result);

                    if (!isApproved)
                    {
                        // User's approval has been revoked, auto-logout
                        await PerformAutoLogout();
                    }
                }
                else
                {
                    // User not found in database, logout
                    await PerformAutoLogout();
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error checking user approval status: {ex.Message}");
            }
        }

        private async Task PerformAutoLogout()
        {
            try
            {
                // Stop timers
                _refreshTimer?.Stop();
                _approvalCheckTimer?.Stop();

                // Clear stored user data
                Preferences.Remove("LoggedInUser");

                // Show alert and navigate to login
                await DisplayAlert("Access Revoked",
                    "Your account access has been revoked by the administrator. You will be logged out.",
                    "OK");

                await Navigation.PushAsync(new UserLoginPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error during auto-logout: {ex.Message}");
            }
        }

        private async System.Threading.Tasks.Task LoadNotificationsAsync()
        {
            try
            {
                var newFrames = new List<Frame>();
                var newNotificationTexts = new ObservableCollection<string>();

                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                string query = "SELECT notify, timestamp FROM logs ORDER BY timestamp DESC LIMIT 10";
                using var cmd = new MySqlCommand(query, conn);
                using var reader = await cmd.ExecuteReaderAsync();

                int index = 0;
                while (await reader.ReadAsync())
                {
                    string notify = reader.GetString(0);
                    DateTime timestamp = reader.GetDateTime(1);
                    string formattedText = $"{timestamp:yyyy-MM-dd HH:mm:ss} - {notify}";

                    // Determine color based on message content
                    Color textColor = Colors.Black;
                    string notifyLower = notify.ToLower();

                    if (notifyLower.Contains("granted"))
                        textColor = Colors.Green;
                    else if (notifyLower.Contains("registered"))
                        textColor = Colors.Green;
                    else if (notifyLower.Contains("denied"))
                        textColor = Colors.Red;
                    else if (notifyLower.Contains("failed"))
                        textColor = Colors.Red;

                    // Create label
                    var label = new Label
                    {
                        Text = formattedText,
                        TextColor = textColor,
                        FontSize = 14,
                        Padding = new Thickness(10, 8)
                    };

                    // Create frame container with special styling for latest notification
                    var frame = new Frame
                    {
                        Content = label,
                        HasShadow = false,
                        Padding = new Thickness(0),
                        Margin = new Thickness(5, 2),
                        CornerRadius = 8
                    };

                    // Highlight the latest notification (first one in the list)
                    if (index == 0)
                    {
                        // Latest notification styling
                        frame.BackgroundColor = Color.FromArgb("#E3F2FD"); // Light blue background
                        frame.BorderColor = Color.FromArgb("#2196F3"); // Blue border
                        frame.HasShadow = true;
                        label.FontAttributes = FontAttributes.Bold;
                        label.FontSize = 15;

                        // Add a "LATEST" indicator
                        var stackLayout = new StackLayout
                        {
                            Orientation = StackOrientation.Horizontal,
                            Children = {
                                new Label
                                {
                                    Text = "🔔 LATEST",
                                    FontSize = 10,
                                    TextColor = Color.FromArgb("#2196F3"),
                                    FontAttributes = FontAttributes.Bold,
                                    VerticalOptions = LayoutOptions.Center
                                },
                                label
                            }
                        };
                        frame.Content = stackLayout;
                    }
                    else
                    {
                        // Regular notification styling
                        frame.BackgroundColor = Colors.White;
                        frame.BorderColor = Color.FromArgb("#E0E0E0");
                    }

                    newFrames.Add(frame);
                    newNotificationTexts.Add(formattedText); // For comparison only
                    index++;
                }

                if (!AreCollectionsEqual(_notifications, newNotificationTexts))
                {
                    // Trigger local notification if new
                    if (newNotificationTexts.Count > 0 &&
                        (_notifications.Count == 0 || _notifications[0] != newNotificationTexts[0]))
                    {
                        await SendLocalNotification(newNotificationTexts[0]);
                    }

                    _notifications = newNotificationTexts;

                    // Update UI
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        NotificationStack.Children.Clear();
                        foreach (var frame in newFrames)
                        {
                            NotificationStack.Children.Add(frame);
                        }
                    });
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading notifications: {ex.Message}");
            }
        }

        private async System.Threading.Tasks.Task SendLocalNotification(string message)
        {
            try
            {
                var request = new NotificationRequest
                {
                    NotificationId = _random.Next(1000, 9999),
                    Title = "New Notification",
                    Description = message
                };

                await LocalNotificationCenter.Current.Show(request);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Notification Error: {ex.Message}");
                // Try with an even simpler notification
                try
                {
                    var simpleRequest = new NotificationRequest
                    {
                        NotificationId = _random.Next(100, 999),
                        Title = "New Alert",
                        Description = "You have a new notification"
                    };
                    await LocalNotificationCenter.Current.Show(simpleRequest);
                }
                catch (Exception innerEx)
                {
                    Console.WriteLine($"Simple Notification Error: {innerEx.Message}");
                }
            }
        }

        private bool AreCollectionsEqual(ObservableCollection<string> a, ObservableCollection<string> b)
        {
            if (a.Count != b.Count) return false;
            for (int i = 0; i < a.Count; i++)
            {
                if (a[i] != b[i]) return false;
            }
            return true;
        }

        // UPDATED: Show Sidebar from the right side - Mobile Compatible
        private async void OnMenuIconTapped(object sender, EventArgs e)
        {
            // Update username before showing sidebar
            UpdateSidebarUsername();

            Sidebar.IsVisible = true;
            Overlay.IsVisible = true;

            // Start from off-screen right
            Sidebar.TranslationX = Sidebar.Width;

            // Animate to visible position
            await Sidebar.TranslateTo(0, 0, 300, Easing.CubicInOut);
        }

        // FIXED: Hide Sidebar to the right side
        private async void OnOverlayTapped(object sender, EventArgs e)
        {
            await HideSidebar();
        }

        // UPDATED: Hide Sidebar - Mobile Compatible  
        private async Task HideSidebar()
        {
            // Animate sidebar to the right (off-screen)
            await Sidebar.TranslateTo(Sidebar.Width, 0, 300, Easing.CubicInOut);
            Sidebar.IsVisible = false;
            Overlay.IsVisible = false;
        }

        // FIXED: Sidebar Button Handlers with proper navigation
        private async void OnApprovalClicked(object sender, EventArgs e)
        {
            try
            {
                await HideSidebar();
                await Navigation.PushAsync(new AdminPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error", $"Could not navigate to Admin page: {ex.Message}", "OK");
            }
        }

        private async void OnDashboardClicked(object sender, EventArgs e)
        {
            try
            {
                await HideSidebar();
                await Navigation.PushAsync(new DPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error", $"Could not navigate to Dashboard: {ex.Message}", "OK");
            }
        }

        private async void OnImageClicked(object sender, EventArgs e)
        {
            try
            {
                await HideSidebar();
                await Navigation.PushAsync(new ImagePage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error", $"Could not navigate to Image page: {ex.Message}", "OK");
            }
        }

        private async void OnLogoutClicked(object sender, EventArgs e)
        {
            try
            {
                bool confirmed = await DisplayAlert("Logout", "Are you sure you want to logout?", "Yes", "No");
                if (confirmed)
                {
                    // Stop timers
                    _refreshTimer?.Stop();
                    _approvalCheckTimer?.Stop();

                    // Clear stored user data
                    Preferences.Remove("LoggedInUser");

                    await HideSidebar();
                    await DisplayAlert("Logout", "Logged out successfully.", "OK");
                    await Navigation.PushAsync(new MainPage());
                    Navigation.RemovePage(this);
                }
                else
                {
                    await HideSidebar();
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Logout Error", $"Could not logout: {ex.Message}", "OK");
            }
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();

            // Stop and dispose timers
            _refreshTimer?.Stop();
            _refreshTimer?.Dispose();
            _approvalCheckTimer?.Stop();
            _approvalCheckTimer?.Dispose();
        }

        private async void OnSolenoidToggled(object sender, ToggledEventArgs e)
        {
            string state = e.Value ? "on" : "off";
            await ToggleSolenoid(state);
        }

        private void OnLiveClicked(object sender, EventArgs e)
        {
            NotificationStack2.IsVisible = false;
            NotificationStack1.IsVisible = false;
            NotificationStack.IsVisible = false;
            feed.IsVisible = true;
        }

        private void OnCloseVideoClicked(object sender, EventArgs e)
        {
            NotificationStack2.IsVisible = true;
            NotificationStack1.IsVisible = true;
            NotificationStack.IsVisible = true;
            feed.IsVisible = false;
        }

        private async void OnAllNotifClicked(object sender, EventArgs e)
        {
            await Navigation.PushAsync(new AllNotifPage());
            Navigation.RemovePage(this);
        }

        private async Task HideApprovalOptionIfUser()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(_loggedInUser))
                    return;

                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                string query = "SELECT status FROM users WHERE LOWER(username) = LOWER(@username)";
                using var cmd = new MySqlCommand(query, conn);
                cmd.Parameters.AddWithValue("@username", _loggedInUser);

                var result = await cmd.ExecuteScalarAsync();

                if (result != null)
                {
                    string status = result.ToString()?.ToLower();

                    if (status == "user")
                    {
                        MainThread.BeginInvokeOnMainThread(() =>
                        {
                            ApprovalButton.IsVisible = false;
                            biometric.IsVisible = false;
                            BiometricSwitch.IsVisible = false;
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error hiding approval option: {ex.Message}");
            }
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            _ = HideApprovalOptionIfUser(); // Fire and forget
        }

        private async Task ToggleSolenoid(string state)
        {
            try
            {
                var httpClient = new HttpClient();
                var content = new FormUrlEncodedContent(new[]
                {
                    new KeyValuePair<string, string>("switch", state)
                });

                // 🔁 Update this with your actual Flask server address
                var response = await httpClient.PostAsync("http://192.168.137.135:5000/control_solenoid", content);
                var responseString = await response.Content.ReadAsStringAsync();
            }
            catch (Exception ex)
            {
                await DisplayAlert("Exception", ex.Message, "OK");
            }
        }

        private void OnSettingsHeaderTapped(object sender, EventArgs e)
        {
            isSettingsExpanded = !isSettingsExpanded;
            SettingsDropdown.IsVisible = isSettingsExpanded;
            SettingsArrow.Source = isSettingsExpanded ? "arrow_up.png" : "arrow_down.png";
        }

        private void OnDarkModeToggled(object sender, ToggledEventArgs e)
        {
            bool isDarkModeOn = e.Value;
            ThemeManager.SetDarkMode(isDarkModeOn, this);
        }

        private void LoadSettings()
        {
            _isLoadingSettings = true; // Set flag to prevent event triggers

            bool darkMode = Preferences.Get("DarkMode", false);
            DarkModeSwitch.IsToggled = darkMode;

            bool biometric = Preferences.Get("BiometricEnabled", false);
            BiometricSwitch.IsToggled = biometric;

            ThemeManager.ApplyTheme(this);

            _isLoadingSettings = false; // Reset flag after loading
        }

        private async void OnIntercomClicked(object sender, EventArgs e)
        {
            // TODO: Implement audio recording and sending to Raspberry Pi via socket
            await DisplayAlert("Intercom", "Intercom button pressed. Implement audio send logic here.", "OK");
        }

        private async void OnBiometricToggled(object sender, ToggledEventArgs e)
        {
            // Prevent dialog from showing when loading settings
            if (_isLoadingSettings)
            {
                return;
            }

            bool isBiometricEnabled = e.Value;
            System.Diagnostics.Debug.WriteLine($"Toggle switched to: {isBiometricEnabled}");

            if (isBiometricEnabled)
            {
                // Create AdminPagelogin instance to access biometric methods
                var adminLoginPage = new AdminPagelogin();

                // First check if biometric authentication is available
                bool isAvailable = await adminLoginPage.IsBiometricAvailableAsync();
                if (!isAvailable)
                {
                    await DisplayAlert("Not Available",
                                     "Biometric authentication is not available on this device.",
                                     "OK");
                    ((Switch)sender).IsToggled = false;
                    return;
                }

                bool userConfirmed = await DisplayAlert("Enable Biometric",
                                                      "Biometric authentication will be enabled for this app. Please authenticate to confirm.",
                                                      "OK", "Cancel");
                if (!userConfirmed)
                {
                    ((Switch)sender).IsToggled = false;
                    return;
                }

                // Use the AdminPagelogin instance to authenticate
                bool biometricSuccess = await adminLoginPage.TriggerBiometricAuthenticationAsync();
                if (!biometricSuccess)
                {
                    await DisplayAlert("Failed",
                                     "Biometric authentication failed or cancelled. Settings not changed.",
                                     "OK");
                    ((Switch)sender).IsToggled = false;
                    return;
                }

                // Save the setting
                Preferences.Set("BiometricEnabled", true);
                bool saved = Preferences.Get("BiometricEnabled", false);
                System.Diagnostics.Debug.WriteLine($"Preference saved as: {saved}");

                await DisplayAlert("Settings",
                                  "Biometric authentication enabled successfully!",
                                  "OK");
            }
            else
            {
                // Save the setting
                Preferences.Set("BiometricEnabled", false);
                bool saved = Preferences.Get("BiometricEnabled", true);
                System.Diagnostics.Debug.WriteLine($"Preference saved as: {saved}");

                await DisplayAlert("Settings",
                                  "Biometric authentication disabled",
                                  "OK");
            }

            // Send message to update login page UI immediately
            MessagingCenter.Send<object>(this, "BiometricSettingsChanged");
        }

        private async void OnIntercomPressed(object sender, EventArgs e)
        {
            try
            {
                if (!await EnsureMicrophonePermissionAsync())
                {
                    await DisplayAlert("Intercom Error", "Microphone permission is required to use the intercom.", "OK");
                    return;
                }

                if (_audioRecorder == null)
                {
                    await DisplayAlert("Intercom Error", "Audio recorder is not initialized.", "OK");
                    return;
                }

                // Notify Pi that button is pressed
                await _socket.EmitAsync("intercom_button", new { pressed = true });

                IntercomButton.Text = "Talking...";
                _isIntercomStreaming = true;
                _intercomCts = new CancellationTokenSource();

                await _audioRecorder.StartAsync(new AudioRecorderOptions
                {
                    SampleRate = 16000,
                    Channels = ChannelType.Mono,
                    BitDepth = BitDepth.Pcm16bit,
                    Encoding = Encoding.LinearPCM,
                });

                // Start streaming audio chunks
                _ = StreamIntercomAudioAsync(_intercomCts.Token);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Intercom Error", ex.Message, "OK");
            }
        }

        private async void OnIntercomReleased(object sender, EventArgs e)
        {
            try
            {
                // Notify Pi that button is released
                await _socket.EmitAsync("intercom_button", new { pressed = false });

                _isIntercomStreaming = false;
                _intercomCts?.Cancel();

                IntercomButton.Text = "Hold to Talk";

                if (_audioRecorder != null)
                {
                    await _audioRecorder.StopAsync();
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Intercom Error", ex.Message, "OK");
            }
        }

        // Helper method to stream audio chunks while button is pressed
        private async Task StreamIntercomAudioAsync(CancellationToken token)
        {
            try
            {
                while (_isIntercomStreaming && !token.IsCancellationRequested)
                {
                    // FIX: Use StopAsync to get the recorded audio, then send it
                    if (_audioRecorder.IsRecording)
                    {
                        var audioSource = await _audioRecorder.StopAsync();
                        if (audioSource != null)
                        {
                            using (var audioStream = audioSource.GetAudioStream())
                            {
                                var buffer = new byte[audioStream.Length];
                                int bytesRead = await audioStream.ReadAsync(buffer, 0, buffer.Length, token);
                                if (bytesRead > 0)
                                {
                                    byte[] chunk = buffer.Take(bytesRead).ToArray();
                                    var audioBase64 = Convert.ToBase64String(chunk);

                                    await _socket.EmitAsync("intercom_stream", new
                                    {
                                        audio = audioBase64,
                                        sample_rate = 16000
                                    });
                                }
                            }
                        }
                        // Restart recording for next chunk
                        await _audioRecorder.StartAsync(new AudioRecorderOptions
                        {
                            SampleRate = 16000,
                            Channels = ChannelType.Mono,
                            BitDepth = BitDepth.Pcm16bit,
                            Encoding = Encoding.LinearPCM,
                        });
                    }
                    await Task.Delay(500, token); // Send every 0.5 seconds
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when cancelled
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Intercom streaming error: {ex.Message}");
            }
        }

        private async Task PlayAudioBytes(byte[] audioBytes, int sampleRate)
        {
            var tempFile = Path.Combine(FileSystem.CacheDirectory, $"intercom_{DateTime.Now.Ticks}.wav");
            await File.WriteAllBytesAsync(tempFile, audioBytes);
            var player = AudioManager.Current.CreateAsyncPlayer(tempFile);
            await player.PlayAsync(CancellationToken.None);
            _ = Task.Delay(5000).ContinueWith(t =>
            {
                player?.Dispose();
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
            });
        }

        private async Task<bool> EnsureMicrophonePermissionAsync()
        {
            var status = await Permissions.CheckStatusAsync<Permissions.Microphone>();
            if (status != PermissionStatus.Granted)
            {
                status = await Permissions.RequestAsync<Permissions.Microphone>();
            }
            return status == PermissionStatus.Granted;
        }
    }
}