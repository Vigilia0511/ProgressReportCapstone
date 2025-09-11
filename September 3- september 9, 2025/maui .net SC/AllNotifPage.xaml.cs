using System;
using System.Collections.Generic;
using Microsoft.Maui.Controls;
using MySql.Data.MySqlClient;
using System.Threading.Tasks;
using System.Timers;
using System.Net.Http;
using Microsoft.Maui.Dispatching;
using testdatabase.Helpers;  // <-- Add this to access ThemeManager

namespace testdatabase
{
    public partial class AllNotifPage : ContentPage
    {
        private readonly Dictionary<Label, int> badgeCountDict = new();
        private readonly string _connStr = "server=10.83.177.213;user=root;password=oneinamillion;database=smartdb;";
        private System.Timers.Timer _pollingTimer;
        private DateTime _lastFetched = DateTime.MinValue;
        private bool _isLoading = false;
        private string _loggedInUser = string.Empty;
        private bool isSettingsExpanded = false;
        private bool _isLoadingSettings = false; // Add this flag to prevent event triggers during loading

        public AllNotifPage()
        {
            InitializeComponent();

            LoadSettings();

            // Get logged in user
            _loggedInUser = Preferences.Get("LoggedInUser", string.Empty);

            // Update sidebar with username
            UpdateSidebarUsername();

            // Initialize badge counts and remove background colors
            badgeCountDict[VoiceBadge] = 0;
            badgeCountDict[FingerprintBadge] = 0;
            badgeCountDict[FaceBadge] = 0;
            badgeCountDict[PINBadge] = 0;

            // Remove background colors from all badges
            VoiceBadge.BackgroundColor = Colors.Transparent;
            FingerprintBadge.BackgroundColor = Colors.Transparent;
            FaceBadge.BackgroundColor = Colors.Transparent;
            PINBadge.BackgroundColor = Colors.Transparent;

            LoadNotifications();

            _pollingTimer = new System.Timers.Timer(2000); // Reduced to 2 seconds for better responsiveness
            _pollingTimer.Elapsed += async (s, e) =>
            {
                await MainThread.InvokeOnMainThreadAsync(LoadNotifications);
            };
            _pollingTimer.AutoReset = true;
            _pollingTimer.Start();

            Shell.SetBackButtonBehavior(this, new BackButtonBehavior
            {
                IsVisible = false,
                IsEnabled = false
            });
        }

        private async void LoadNotifications()
        {
            if (_isLoading) return; // Prevent multiple simultaneous loads
            _isLoading = true;

            try
            {
                var voiceEntries = new List<(string message, bool denied, DateTime timestamp)>();
                var fingerprintEntries = new List<(string message, bool denied, DateTime timestamp)>();
                var faceEntries = new List<(string message, bool denied, DateTime timestamp)>();
                var pinEntries = new List<(string message, bool denied, DateTime timestamp)>();

                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                string query = "SELECT notify, timestamp FROM logs WHERE timestamp > @lastFetched ORDER BY timestamp ASC";
                using var cmd = new MySqlCommand(query, conn);
                cmd.Parameters.AddWithValue("@lastFetched", _lastFetched);
                using var reader = await cmd.ExecuteReaderAsync();

                var newMaxTimestamp = _lastFetched;

                while (await reader.ReadAsync())
                {
                    string notify = reader.GetString(0);
                    DateTime timestamp = reader.GetDateTime(1);

                    // Update max timestamp
                    if (timestamp > newMaxTimestamp)
                        newMaxTimestamp = timestamp;

                    bool denied = notify.ToLower().Contains("denied") || notify.ToLower().Contains("failed");

                    if (notify.ToLower().Contains("voice"))
                        voiceEntries.Add((notify, denied, timestamp));
                    else if (notify.ToLower().Contains("fingerprint"))
                        fingerprintEntries.Add((notify, denied, timestamp));
                    else if (notify.ToLower().Contains("face"))
                        faceEntries.Add((notify, denied, timestamp));
                    else if (notify.ToLower().Contains("pin"))
                        pinEntries.Add((notify, denied, timestamp));
                }

                // Update timestamp only after successful fetch
                _lastFetched = newMaxTimestamp;

                // Add notifications in parallel for better performance
                var tasks = new List<Task>
                {
                    AddNotificationsAsync(VoiceStack, VoiceBadge, voiceEntries),
                    AddNotificationsAsync(FingerprintStack, FingerprintBadge, fingerprintEntries),
                    AddNotificationsAsync(FaceStack, FaceBadge, faceEntries),
                    AddNotificationsAsync(PINStack, PINBadge, pinEntries)
                };

                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", $"Failed to load logs: {ex.Message}", "OK");
            }
            finally
            {
                _isLoading = false;
            }
        }

        private async Task AddNotificationsAsync(VerticalStackLayout stack, Label badge, List<(string message, bool denied, DateTime timestamp)> entries)
        {
            if (entries.Count == 0) return;

            // Process entries in batches for better UI responsiveness
            const int batchSize = 30;
            for (int i = 0; i < entries.Count; i += batchSize)
            {
                var batch = entries.Skip(i).Take(batchSize);
                var tasks = batch.Select(entry => AddNotificationAsync(stack, badge, entry.message, entry.denied, entry.timestamp));
                await Task.WhenAll(tasks);

                // Small delay between batches to keep UI responsive
                if (i + batchSize < entries.Count)
                    await Task.Delay(10);
            }
        }

        private async Task AddNotificationAsync(VerticalStackLayout stack, Label badge, string message, bool isDenied, DateTime timestamp)
        {
            // Create the main notification frame with minimal spacing
            var mainFrame = new Frame
            {
                BorderColor = Colors.LightGray,
                BackgroundColor = Colors.White,
                CornerRadius = 8, // Reduced corner radius for tighter look
                Margin = new Thickness(0, 0), // Removed all margins for tightest spacing
                Padding = new Thickness(6, 4), // Further reduced padding
                HasShadow = false // Removed shadow for cleaner look
            };

            // Create the content grid - single column layout
            var contentGrid = new Grid
            {
                ColumnDefinitions = { new ColumnDefinition(GridLength.Star) },
                RowDefinitions = { new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Auto) },
                RowSpacing = 1 // Even tighter spacing between message and timestamp
            };

            // Main message text with color coding
            var messageLabel = new Label
            {
                Text = message,
                FontSize = 13, // Slightly smaller font
                FontAttributes = FontAttributes.Bold,
                TextColor = isDenied ? Color.FromArgb("#D32F2F") : Color.FromArgb("#2E7D32"),
                VerticalOptions = LayoutOptions.Start,
                Margin = new Thickness(0), // Remove any default margins
                LineBreakMode = LineBreakMode.TailTruncation // Prevent text wrapping
            };

            // Timestamp label
            var timestampLabel = new Label
            {
                Text = GetRelativeTime(timestamp),
                FontSize = 10, // Smaller timestamp
                TextColor = Colors.Gray,
                VerticalOptions = LayoutOptions.Start,
                Margin = new Thickness(0) // Remove any default margins
            };

            // Add tap gesture for interaction
            var tapGesture = new TapGestureRecognizer();
            tapGesture.Tapped += async (s, e) => await OnNotificationTapped(message, timestamp, isDenied);
            mainFrame.GestureRecognizers.Add(tapGesture);

            // Layout the content - simplified single column
            contentGrid.Add(messageLabel, 0, 0);
            contentGrid.Add(timestampLabel, 0, 1);

            mainFrame.Content = contentGrid;

            // Add without heavy animations for better performance
            mainFrame.Opacity = 0.8;
            stack.Children.Insert(0, mainFrame); // Insert at top (latest first)

            // Quick fade in
            await mainFrame.FadeTo(1, 150, Easing.CubicOut);

            // Update badge count and remove background color
            badgeCountDict[badge]++;
            badge.Text = badgeCountDict[badge].ToString();
            badge.BackgroundColor = Colors.Transparent; // Remove background color

            // Auto-remove old notifications (keep only last 15 for better performance)
            if (stack.Children.Count > 100000)
            {
                var oldNotification = stack.Children[stack.Children.Count - 1];
                stack.Children.Remove(oldNotification);
            }
        }

        private async Task AnimateBadgeUpdate(Label badge, int count)
        {
            badge.Text = count.ToString();
            await badge.ScaleTo(1.2, 100, Easing.CubicOut);
            await badge.ScaleTo(1.0, 100, Easing.CubicIn);
        }

        private string GetRelativeTime(DateTime timestamp)
        {
            var timeSpan = DateTime.Now - timestamp;

            if (timeSpan.TotalMinutes < 1)
                return "Just now";
            else if (timeSpan.TotalMinutes < 60)
                return $"{(int)timeSpan.TotalMinutes}m ago";
            else if (timeSpan.TotalHours < 24)
                return $"{(int)timeSpan.TotalHours}h ago";
            else if (timeSpan.TotalDays < 7)
                return $"{(int)timeSpan.TotalDays}d ago";
            else
                return timestamp.ToString("MMM dd");
        }

        private async Task OnNotificationTapped(string message, DateTime timestamp, bool isDenied)
        {
            string status = isDenied ? "DENIED" : "APPROVED";
            string details = $"Status: {status}\n" +
                           $"Message: {message}\n" +
                           $"Time: {timestamp:yyyy-MM-dd HH:mm:ss}\n" +
                           $"Relative: {GetRelativeTime(timestamp)}";

            await DisplayAlert("Notification Details", details, "OK");
        }

        // Add method to clear notifications for a specific category
        private async void OnClearCategoryClicked(VerticalStackLayout stack, Label badge, string categoryName)
        {
            bool confirm = await DisplayAlert("Clear Notifications",
                $"Are you sure you want to clear all {categoryName} notifications?",
                "Yes", "No");

            if (confirm)
            {
                // Animate out all notifications
                var tasks = new List<Task>();
                var childrenToRemove = new List<View>();

                foreach (View child in stack.Children)
                {
                    childrenToRemove.Add(child);
                    // Cast View to VisualElement to access FadeTo
                    if (child is VisualElement visualElement)
                    {
                        tasks.Add(visualElement.FadeTo(0, 200));
                    }
                }

                await Task.WhenAll(tasks);

                foreach (var child in childrenToRemove)
                {
                    stack.Children.Remove(child);
                }

                badgeCountDict[badge] = 0;
                badge.Text = "0";
                badge.BackgroundColor = Colors.Transparent; // Ensure background stays transparent

                await DisplayAlert("Success", $"{categoryName} notifications cleared.", "OK");
            }
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            _pollingTimer?.Stop();
        }

        // UPDATED: Show Sidebar from the right side - Mobile Compatible
        private async void OnMenuIconTapped(object sender, EventArgs e)
        {
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

        private async void OnCloseNotifClicked(object sender, EventArgs e)
        {
            try
            {
                await Navigation.PushAsync(new DPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error", $"Could not navigate: {ex.Message}", "OK");
            }
        }

        private void OnCloseVideoClicked(object sender, EventArgs e)
        {
            AllNotif.IsVisible = true;
            closenotif.IsVisible = true;
            feed.IsVisible = false;
        }

        private void OnLiveClicked(object sender, EventArgs e)
        {
            AllNotif.IsVisible = false;
            feed.IsVisible = true;
            closenotif.IsVisible = false;
        }

        private async void OnSolenoidToggled(object sender, ToggledEventArgs e)
        {
            string state = e.Value ? "on" : "off";
            await ToggleSolenoid(state);
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

                var response = await httpClient.PostAsync("http://192.168.176.237:5000/control_solenoid", content);
                var responseString = await response.Content.ReadAsStringAsync();
            }
            catch (Exception ex)
            {
                await DisplayAlert("Exception", ex.Message, "OK");
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
    }
}