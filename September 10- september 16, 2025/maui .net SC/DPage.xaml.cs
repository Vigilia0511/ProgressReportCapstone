using Plugin.LocalNotification;
using System;
using System.Collections.ObjectModel;
using Microsoft.Maui.Controls;
using MySql.Data.MySqlClient;
using System.Timers;
using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Storage;
using testdatabase.Helpers;
using Plugin.Maui.Audio;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;

namespace testdatabase
{
    public partial class DPage : ContentPage
    {
        private bool _isSidebarVisible = false;
        private System.Timers.Timer _refreshTimer;
        private System.Timers.Timer _approvalCheckTimer;
        private ObservableCollection<string> _notifications = new();
        private readonly string _connStr = "server=192.168.1.149;user=root;password=oneinamillion;database=smartdb;";
        private Random _random = new Random();
        private string _loggedInUser = string.Empty;
        private bool isSettingsExpanded = false;
        private bool _isLoadingSettings = false;
        private IAudioRecorder _audioRecorder;
        private UdpClient _udpClient;
        private IPEndPoint _udpEndPoint;
        private bool _isIntercomStreaming = false;
        private CancellationTokenSource _intercomCts;
        private bool _isRecording = false;
        private bool _isIntercomToggled = false; // For Windows toggle

        private System.Timers.Timer _videoFeedRefreshTimer;
        private const int VideoFeedRefreshIntervalMs = 5000; // Refresh every 5 seconds

        public DPage()
        {
            InitializeComponent();

            // Bind event handlers for IntercomButton with null check
            if (IntercomButton != null)
            {
#if WINDOWS
                IntercomButton.Clicked += OnIntercomClickedWindows; // Toggle for Windows
#else
                IntercomButton.Pressed += OnIntercomPressed;
                IntercomButton.Released += OnIntercomReleased;
#endif
            }
            else
                System.Diagnostics.Debug.WriteLine("IntercomButton not found in XAML");

            LoadSettings();
            Shell.SetBackButtonBehavior(this, new BackButtonBehavior { IsVisible = false, IsEnabled = false });

            _loggedInUser = Preferences.Get("LoggedInUser", string.Empty);
            UpdateSidebarUsername();

            LoadNotificationsAsync();

            _refreshTimer = new System.Timers.Timer(10000);
            _refreshTimer.Elapsed += RefreshTimer_Elapsed;
            _refreshTimer.AutoReset = true;
            _refreshTimer.Start();

            _approvalCheckTimer = new System.Timers.Timer(15000);
            _approvalCheckTimer.Elapsed += ApprovalCheckTimer_Elapsed;
            _approvalCheckTimer.AutoReset = true;
            _approvalCheckTimer.Start();

            _audioRecorder = AudioManager.Current.CreateRecorder();

            _udpEndPoint = new IPEndPoint(IPAddress.Parse("192.168.1.117"), 5005);
            _udpClient = new UdpClient();
        }

        private void UpdateSidebarUsername()
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(_loggedInUser) && SidebarUsernameLabel != null)
                {
                    SidebarUsernameLabel.Text = $" {_loggedInUser}";
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"SidebarUsernameLabel is null or username is empty: {_loggedInUser}");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating sidebar username: {ex.Message}");
            }
        }

        private void RefreshTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            MainThread.BeginInvokeOnMainThread(async () => await LoadNotificationsAsync());
        }

        private void ApprovalCheckTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            MainThread.BeginInvokeOnMainThread(async () => await CheckUserApprovalStatus());
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
                if (result != null && !Convert.ToBoolean(result))
                {
                    await PerformAutoLogout();
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error checking user approval: {ex.Message}");
            }
        }

        private async Task PerformAutoLogout()
        {
            try
            {
                _refreshTimer?.Stop();
                _approvalCheckTimer?.Stop();
                Preferences.Remove("LoggedInUser");

                await DisplayAlert("Access Revoked", "Your account access has been revoked.", "OK");
                await Navigation.PushAsync(new UserLoginPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error during auto-logout: {ex.Message}");
            }
        }

        private async Task LoadNotificationsAsync()
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

                    Color textColor = Colors.Black;
                    string notifyLower = notify.ToLower();
                    if (notifyLower.Contains("granted") || notifyLower.Contains("registered"))
                        textColor = Colors.Green;
                    else if (notifyLower.Contains("denied") || notifyLower.Contains("failed"))
                        textColor = Colors.Red;

                    var label = new Label
                    {
                        Text = formattedText,
                        TextColor = textColor,
                        FontSize = 14,
                        Padding = new Thickness(10, 8)
                    };

                    var frame = new Frame
                    {
                        HasShadow = false,
                        Padding = new Thickness(0),
                        Margin = new Thickness(5, 2),
                        CornerRadius = 8
                    };

                    if (index == 0)
                    {
                        frame.BackgroundColor = Color.FromArgb("#E3F2FD");
                        frame.BorderColor = Color.FromArgb("#2196F3");
                        frame.HasShadow = true;

                        var latestLabel = new Label
                        {
                            Text = formattedText,
                            TextColor = textColor,
                            FontSize = 15,
                            FontAttributes = FontAttributes.Bold,
                            Padding = new Thickness(10, 8)
                        };

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
                                latestLabel
                            }
                        };
                        frame.Content = stackLayout;
                    }
                    else
                    {
                        frame.BackgroundColor = Colors.White;
                        frame.BorderColor = Color.FromArgb("#E0E0E0");
                        frame.Content = label;
                    }

                    newFrames.Add(frame);
                    newNotificationTexts.Add(formattedText);
                    index++;
                }

                if (!AreCollectionsEqual(_notifications, newNotificationTexts))
                {
                    if (newNotificationTexts.Count > 0 && (_notifications.Count == 0 || _notifications[0] != newNotificationTexts[0]))
                    {
                        await SendLocalNotification(newNotificationTexts[0]);
                    }

                    _notifications = newNotificationTexts;
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        if (NotificationStack != null)
                        {
                            NotificationStack.Children.Clear();
                            foreach (var frame in newFrames)
                            {
                                NotificationStack.Children.Add(frame);
                            }
                        }
                        else
                        {
                            System.Diagnostics.Debug.WriteLine("NotificationStack is null");
                        }
                    });
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading notifications: {ex.Message}");
            }
        }

        private async Task SendLocalNotification(string message)
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
                System.Diagnostics.Debug.WriteLine($"Notification Error: {ex.Message}");
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

        private async void OnMenuIconTapped(object sender, EventArgs e)
        {
            try
            {
                UpdateSidebarUsername();
                if (Sidebar != null && Overlay != null)
                {
                    Sidebar.IsVisible = true;
                    Overlay.IsVisible = true;
                    Sidebar.TranslationX = Sidebar.Width;
                    await Sidebar.TranslateTo(0, 0, 300, Easing.CubicInOut);
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("Sidebar or Overlay is null");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error in OnMenuIconTapped: {ex.Message}");
            }
        }

        private async void OnOverlayTapped(object sender, EventArgs e)
        {
            await HideSidebar();
        }

        private async Task HideSidebar()
        {
            try
            {
                if (Sidebar != null && Overlay != null)
                {
                    await Sidebar.TranslateTo(Sidebar.Width, 0, 300, Easing.CubicInOut);
                    Sidebar.IsVisible = false;
                    Overlay.IsVisible = false;
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("Sidebar or Overlay is null in HideSidebar");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error hiding sidebar: {ex.Message}");
            }
        }

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
                    _refreshTimer?.Stop();
                    _approvalCheckTimer?.Stop();
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
            _refreshTimer?.Stop();
            _refreshTimer?.Dispose();
            _approvalCheckTimer?.Stop();
            _approvalCheckTimer?.Dispose();

            // Stop the video feed refresh timer
            StopVideoFeedAutoRefresh();

            // Force the WebView to disconnect by loading a blank page
            if (VideoFeedWebView != null)
            {
                VideoFeedWebView.Source = "about:blank";
            }

            if (_isRecording)
            {
                try
                {
                    _audioRecorder?.StopAsync();
                    _isRecording = false;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error stopping recording: {ex.Message}");
                }
            }

            _isIntercomStreaming = false;
            _intercomCts?.Cancel();
            _intercomCts?.Dispose();
            _udpClient?.Close();
            _udpClient?.Dispose();
        }

        private async void OnSolenoidToggled(object sender, ToggledEventArgs e)
        {
            string state = e.Value ? "on" : "off";
            await ToggleSolenoid(state);
        }

        private void OnLiveClicked(object sender, EventArgs e)
        {
            try
            {
                if (NotificationStack2 != null) NotificationStack2.IsVisible = false;
                if (NotificationStack1 != null) NotificationStack1.IsVisible = false;
                if (NotificationStack != null) NotificationStack.IsVisible = false;
                if (feed != null) feed.IsVisible = true;

                // Activate the live view only now
                if (VideoFeedWebView != null)
                {
                    VideoFeedWebView.Source = "http://192.168.1.117:5000/video_feed";
                }
                StartVideoFeedAutoRefresh();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error in OnLiveClicked: {ex.Message}");
            }
        }

        private void OnCloseVideoClicked(object sender, EventArgs e)
        {
            try
            {
                if (NotificationStack2 != null) NotificationStack2.IsVisible = true;
                if (NotificationStack1 != null) NotificationStack1.IsVisible = true;
                if (NotificationStack != null) NotificationStack.IsVisible = true;
                if (feed != null) feed.IsVisible = false;

                // Stop the video feed refresh timer
                StopVideoFeedAutoRefresh();

                // Force the WebView to disconnect by loading a blank page
                if (VideoFeedWebView != null)
                {
                    VideoFeedWebView.Source = "about:blank";
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error in OnCloseVideoClicked: {ex.Message}");
            }
        }

        private async void OnAllNotifClicked(object sender, EventArgs e)
        {
            try
            {
                await Navigation.PushAsync(new AllNotifPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error", $"Could not navigate to All Notifications page: {ex.Message}", "OK");
            }
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
                if (result?.ToString()?.ToLower() == "user")
                {
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        if (ApprovalButton != null && biometric != null && BiometricSwitch != null)
                        {
                            ApprovalButton.IsVisible = false;
                            biometric.IsVisible = false;
                            BiometricSwitch.IsVisible = false;
                        }
                        else
                        {
                            System.Diagnostics.Debug.WriteLine("ApprovalButton, biometric, or BiometricSwitch is null");
                        }
                    });
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error hiding approval option: {ex.Message}");
            }
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();
            await HideApprovalOptionIfUser();
        }

        private async Task ToggleSolenoid(string state)
        {
            try
            {
                // TODO: Verify if http://192.168.1.117:5000/control_solenoid is correct or remove if not needed
                var httpClient = new HttpClient();
                var content = new FormUrlEncodedContent(new[]
                {
                    new KeyValuePair<string, string>("switch", state)
                });
                var response = await httpClient.PostAsync("http://192.168.1.117:5000/control_solenoid", content);
                await response.Content.ReadAsStringAsync();
            }
            catch (Exception ex)
            {
                await DisplayAlert("Solenoid Error", $"Failed to toggle solenoid: {ex.Message}", "OK");
            }
        }

        private void OnSettingsHeaderTapped(object sender, EventArgs e)
        {
            try
            {
                if (SettingsDropdown != null && SettingsArrow != null)
                {
                    isSettingsExpanded = !isSettingsExpanded;
                    SettingsDropdown.IsVisible = isSettingsExpanded;
                    SettingsArrow.Source = isSettingsExpanded ? "arrow_up.png" : "arrow_down.png";
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("SettingsDropdown or SettingsArrow is null");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error in OnSettingsHeaderTapped: {ex.Message}");
            }
        }

        private void OnDarkModeToggled(object sender, ToggledEventArgs e)
        {
            bool isDarkModeOn = e.Value;
            ThemeManager.SetDarkMode(isDarkModeOn, this);
        }

        private void LoadSettings()
        {
            _isLoadingSettings = true;
            if (DarkModeSwitch != null)
                DarkModeSwitch.IsToggled = Preferences.Get("DarkMode", false);
            if (BiometricSwitch != null)
                BiometricSwitch.IsToggled = Preferences.Get("BiometricEnabled", false);
            ThemeManager.ApplyTheme(this);
            _isLoadingSettings = false;
        }

        private async void OnIntercomClicked(object sender, EventArgs e)
        {
            await DisplayAlert("Intercom", "Hold the Intercom button to talk (or click to toggle on Windows).", "OK");
        }

        private async void OnBiometricToggled(object sender, ToggledEventArgs e)
        {
            if (_isLoadingSettings)
                return;

            bool isBiometricEnabled = e.Value;
            if (isBiometricEnabled)
            {
                var adminLoginPage = new AdminPagelogin();
                bool isAvailable = await adminLoginPage.IsBiometricAvailableAsync();
                if (!isAvailable)
                {
                    await DisplayAlert("Not Available", "Biometric authentication is not available.", "OK");
                    if (sender is Switch sw) sw.IsToggled = false;
                    return;
                }

                bool userConfirmed = await DisplayAlert("Enable Biometric", "Authenticate to enable biometric authentication.", "OK", "Cancel");
                if (!userConfirmed)
                {
                    if (sender is Switch sw) sw.IsToggled = false;
                    return;
                }

                bool biometricSuccess = await adminLoginPage.TriggerBiometricAuthenticationAsync();
                if (!biometricSuccess)
                {
                    await DisplayAlert("Failed", "Biometric authentication failed or cancelled.", "OK");
                    if (sender is Switch sw) sw.IsToggled = false;
                    return;
                }

                Preferences.Set("BiometricEnabled", true);
                await DisplayAlert("Settings", "Biometric authentication enabled.", "OK");
            }
            else
            {
                Preferences.Set("BiometricEnabled", false);
                await DisplayAlert("Settings", "Biometric authentication disabled.", "OK");
            }

            MessagingCenter.Send<object>(this, "BiometricSettingsChanged");
        }

        private async void OnIntercomPressed(object sender, EventArgs e)
        {
            System.Diagnostics.Debug.WriteLine("Intercom button pressed (mobile)");
            if (!await EnsureMicrophonePermissionAsync())
            {
                await DisplayAlert("Intercom Error", "Microphone permission required.", "OK");
                return;
            }

            if (_audioRecorder == null)
            {
                await DisplayAlert("Intercom Error", "Audio recorder not initialized.", "OK");
                return;
            }

            if (_isRecording)
                return;

            try
            {
                await _audioRecorder.StartAsync(new AudioRecorderOptions
                {
                    SampleRate = 16000,
                    Channels = ChannelType.Mono,
                    BitDepth = BitDepth.Pcm16bit,
                    Encoding = Plugin.Maui.Audio.Encoding.LinearPCM,
                });
                _isRecording = true;
                _isIntercomStreaming = true;
                _intercomCts = new CancellationTokenSource();
                System.Diagnostics.Debug.WriteLine("Starting intercom streaming (mobile)");
                _ = StreamIntercomAudioUdpAsync(_intercomCts.Token);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"AudioRecorder Start Error: {ex.Message}");
                await DisplayAlert("Recording Error", $"Failed to start recording: {ex.Message}", "OK");
            }
        }

        private async void OnIntercomReleased(object sender, EventArgs e)
        {
            System.Diagnostics.Debug.WriteLine("Intercom button released (mobile)");
            _isIntercomStreaming = false;
            _intercomCts?.Cancel();

            if (_isRecording && _audioRecorder != null)
            {
                try
                {
                    await _audioRecorder.StopAsync();
                    _isRecording = false;
                    System.Diagnostics.Debug.WriteLine("Stopped recording (mobile)");
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error stopping audio recorder: {ex.Message}");
                }
            }
        }

        private async void OnIntercomClickedWindows(object sender, EventArgs e)
        {
            System.Diagnostics.Debug.WriteLine("Intercom button clicked (Windows)");
            if (!await EnsureMicrophonePermissionAsync())
            {
                await DisplayAlert("Intercom Error", "Microphone permission required.", "OK");
                return;
            }

            if (_audioRecorder == null)
            {
                await DisplayAlert("Intercom Error", "Audio recorder not initialized.", "OK");
                return;
            }

            _isIntercomToggled = !_isIntercomToggled; // Toggle state
            System.Diagnostics.Debug.WriteLine($"Intercom toggled: {_isIntercomToggled}");

            if (_isIntercomToggled)
            {
                if (!_isRecording)
                {
                    try
                    {
                        await _audioRecorder.StartAsync(new AudioRecorderOptions
                        {
                            SampleRate = 16000,
                            Channels = ChannelType.Mono,
                            BitDepth = BitDepth.Pcm16bit,
                            Encoding = Plugin.Maui.Audio.Encoding.LinearPCM,
                        });
                        _isRecording = true;
                        _isIntercomStreaming = true;
                        _intercomCts = new CancellationTokenSource();
                        System.Diagnostics.Debug.WriteLine("Starting intercom streaming (Windows)");
                        _ = StreamIntercomAudioUdpAsync(_intercomCts.Token);
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"AudioRecorder Start Error (Windows): {ex.Message}");
                        await DisplayAlert("Recording Error", $"Failed to start recording: {ex.Message}", "OK");
                    }
                }
            }
            else
            {
                _isIntercomStreaming = false;
                _intercomCts?.Cancel();

                if (_isRecording && _audioRecorder != null)
                {
                    try
                    {
                        await _audioRecorder.StopAsync();
                        _isRecording = false;
                        System.Diagnostics.Debug.WriteLine("Stopped recording (Windows)");
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Error stopping audio recorder (Windows): {ex.Message}");
                    }
                }
            }
        }

        private async Task StreamIntercomAudioUdpAsync(CancellationToken token)
        {
            const int chunkDurationMs = 500; // Record 500ms chunks
            const int sampleRate = 16000;
            const int bytesPerSample = 2; // PCM16, mono
            const int chunkSize = sampleRate * bytesPerSample * chunkDurationMs / 1000; // 16000 bytes for 0.5s
            byte[] buffer = new byte[chunkSize];
            int bufferOffset = 0;

            try
            {
                while (_isIntercomStreaming && !token.IsCancellationRequested && _isRecording)
                {
                    if (!_audioRecorder.IsRecording)
                    {
                        System.Diagnostics.Debug.WriteLine("Recorder stopped unexpectedly.");
                        break;
                    }

                    // Record for chunkDurationMs
                    await Task.Delay(chunkDurationMs, token);
                    var audioSource = await _audioRecorder.StopAsync();
                    _isRecording = false;

                    if (audioSource != null)
                    {
                        using var audioStream = audioSource.GetAudioStream();
                        using var ms = new MemoryStream();
                        await audioStream.CopyToAsync(ms, token);
                        var allBytes = ms.ToArray();
                        System.Diagnostics.Debug.WriteLine($"Captured {allBytes.Length} bytes of audio data");

                        if (allBytes.Length > 0)
                        {
                            var pcm = ExtractPcmFromWav(allBytes);
                            System.Diagnostics.Debug.WriteLine($"Extracted PCM: {pcm?.Length ?? 0} bytes");

                            if (pcm != null && pcm.Length > 0)
                            {
                                int pcmOffset = 0;
                                while (pcmOffset < pcm.Length && !token.IsCancellationRequested)
                                {
                                    int bytesToCopy = Math.Min(chunkSize - bufferOffset, pcm.Length - pcmOffset);
                                    Array.Copy(pcm, pcmOffset, buffer, bufferOffset, bytesToCopy);
                                    bufferOffset += bytesToCopy;
                                    pcmOffset += bytesToCopy;

                                    if (bufferOffset >= chunkSize)
                                    {
                                        await _udpClient.SendAsync(buffer, chunkSize, _udpEndPoint);
                                        System.Diagnostics.Debug.WriteLine($"Sent {chunkSize} bytes to UDP {_udpEndPoint}");
                                        bufferOffset = 0;
                                        Array.Clear(buffer, 0, buffer.Length); // Clear buffer for next chunk
                                    }
                                }
                            }
                        }
                    }

                    if (_isIntercomStreaming && !token.IsCancellationRequested)
                    {
                        await _audioRecorder.StartAsync(new AudioRecorderOptions
                        {
                            SampleRate = 16000,
                            Channels = ChannelType.Mono,
                            BitDepth = BitDepth.Pcm16bit,
                            Encoding = Plugin.Maui.Audio.Encoding.LinearPCM,
                        });
                        _isRecording = true;
                    }
                }

                // Send any remaining data in buffer
                if (bufferOffset > 0 && !token.IsCancellationRequested)
                {
                    await _udpClient.SendAsync(buffer, bufferOffset, _udpEndPoint);
                    System.Diagnostics.Debug.WriteLine($"Sent remaining {bufferOffset} bytes to UDP {_udpEndPoint}");
                }
            }
            catch (OperationCanceledException)
            {
                System.Diagnostics.Debug.WriteLine("Audio streaming cancelled.");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Intercom streaming error: {ex.Message}");
            }
        }

        private byte[] ExtractPcmFromWav(byte[] wav)
        {
            if (wav == null || wav.Length < 44)
            {
                System.Diagnostics.Debug.WriteLine("WAV data too short or null");
                return wav ?? Array.Empty<byte>();
            }

            for (int i = 0; i < wav.Length - 8; i++)
            {
                if (wav[i] == (byte)'d' && wav[i + 1] == (byte)'a' && wav[i + 2] == (byte)'t' && wav[i + 3] == (byte)'a')
                {
                    int dataSize = BitConverter.ToInt32(wav, i + 4);
                    int start = i + 8;
                    if (start + dataSize <= wav.Length)
                    {
                        var pcm = new byte[dataSize];
                        Buffer.BlockCopy(wav, start, pcm, 0, dataSize);
                        System.Diagnostics.Debug.WriteLine($"Extracted PCM data: {dataSize} bytes");
                        return pcm;
                    }
                    System.Diagnostics.Debug.WriteLine("Invalid WAV data size");
                    break;
                }
            }

            System.Diagnostics.Debug.WriteLine("No 'data' chunk found, skipping WAV header");
            return wav.Length > 44 ? wav.Skip(44).ToArray() : Array.Empty<byte>();
        }

        private async Task<bool> EnsureMicrophonePermissionAsync()
        {
            try
            {
                var status = await Permissions.CheckStatusAsync<Permissions.Microphone>();
                if (status != PermissionStatus.Granted)
                {
                    status = await Permissions.RequestAsync<Permissions.Microphone>();
                }
                System.Diagnostics.Debug.WriteLine($"Microphone permission status: {status}");
                return status == PermissionStatus.Granted;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error checking microphone permissions: {ex.Message}");
                return false;
            }
        }

        private void StartVideoFeedAutoRefresh()
        {
            if (_videoFeedRefreshTimer == null)
            {
                _videoFeedRefreshTimer = new System.Timers. Timer(VideoFeedRefreshIntervalMs);
                _videoFeedRefreshTimer.Elapsed += (s, e) =>
                {
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        if (feed != null && feed.IsVisible && VideoFeedWebView != null)
                        {
                            // Reload the WebView source to refresh the feed
                            VideoFeedWebView.Source = null;
                            VideoFeedWebView.Source = "http://192.168.1.117:5000/video_feed";
                        }
                    });
                };
                _videoFeedRefreshTimer.AutoReset = true;
            }
            _videoFeedRefreshTimer.Start();
        }

        private void StopVideoFeedAutoRefresh()
        {
            _videoFeedRefreshTimer?.Stop();
        }
    }
}