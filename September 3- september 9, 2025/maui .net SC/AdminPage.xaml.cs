using MySql.Data.MySqlClient;
using System.Collections.ObjectModel;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
using System.Timers;
using testdatabase.Helpers;  // <-- Add this to access ThemeManager


namespace testdatabase
{
    public partial class AdminPage : ContentPage
    {
        private bool isSidebarVisible = false;
        private readonly string connStr = "server=10.83.177.213;user=root;password=oneinamillion;database=smartdb;";
        private ObservableCollection<UserModel> userList = new();
        private System.Timers.Timer refreshTimer;
        private bool isSettingsExpanded = false;
        private bool _isLoadingSettings = false; // Add this flag to prevent event triggers during loading

        public AdminPage()
        {
            InitializeComponent();
            LoadUsers();
            LoadSettings();

            Shell.SetBackButtonBehavior(this, new BackButtonBehavior
            {
                IsVisible = false,
                IsEnabled = false
            });

            refreshTimer = new System.Timers.Timer(5000);
            refreshTimer.Elapsed += (s, e) =>
            {
                MainThread.BeginInvokeOnMainThread(() => LoadUsers());
            };
            refreshTimer.AutoReset = true;
            refreshTimer.Start();
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            ThemeManager.ApplyTheme(this);
        }


        private async void LoadUsers()
        {
            try
            {
                var latestUsers = new ObservableCollection<UserModel>();

                using var conn = new MySqlConnection(connStr);
                await conn.OpenAsync();

                string query = "SELECT id, username, is_approved FROM users ORDER BY id DESC";
                using var cmd = new MySqlCommand(query, conn);
                using var reader = (MySqlDataReader)await cmd.ExecuteReaderAsync();

                while (await reader.ReadAsync())
                {
                    latestUsers.Add(new UserModel
                    {
                        Id = reader.GetInt32("id"),
                        Username = reader.GetString("username"),
                        IsApproved = reader.GetBoolean("is_approved")
                    });
                }

                if (!AreCollectionsEqual(userList, latestUsers))
                {
                    userList = latestUsers;
                    userCollectionView.ItemsSource = null;
                    userCollectionView.ItemsSource = userList;
                }
            }
            catch (Exception ex)
            {
                // Avoid spamming alerts every refresh
                System.Diagnostics.Debug.WriteLine($"LoadUsers error: {ex.Message}");
            }
        }

        private bool AreCollectionsEqual(ObservableCollection<UserModel> a, ObservableCollection<UserModel> b)
        {
            if (a.Count != b.Count) return false;
            for (int i = 0; i < a.Count; i++)
            {
                if (a[i].Id != b[i].Id ||
                    a[i].Username != b[i].Username ||
                    a[i].IsApproved != b[i].IsApproved)
                    return false;
            }
            return true;
        }

        private async void OnToggleApprovalClicked(object sender, EventArgs e)
        {
            if (sender is Button button && button.CommandParameter is UserModel user)
            {
                try
                {
                    bool newApprovalValue = !user.IsApproved;

                    using var conn = new MySqlConnection(connStr);
                    await conn.OpenAsync();

                    string updateQuery = "UPDATE users SET is_approved = @approve WHERE username = @username";
                    using var cmd = new MySqlCommand(updateQuery, conn);
                    cmd.Parameters.AddWithValue("@approve", newApprovalValue);
                    cmd.Parameters.AddWithValue("@username", user.Username);
                    await cmd.ExecuteNonQueryAsync();

                    // Update the list locally
                    user.IsApproved = newApprovalValue;

                    // Refresh UI
                    userCollectionView.ItemsSource = null;
                    userCollectionView.ItemsSource = userList;
                }
                catch (Exception ex)
                {
                    await DisplayAlert("Error", ex.Message, "OK");
                }
            }
        }

        private async void OnLogoutClicked(object sender, EventArgs e)
        {
            bool confirmed = await DisplayAlert("Logout", "Are you sure you want to logout?", "Yes", "No");
            if (confirmed)
            {
                await DisplayAlert("Logout", "Logged out successfully.", "OK");
                await Navigation.PushAsync(new MainPage());
                Navigation.RemovePage(this);
            }
            await HideSidebarAsync();
        }

        private void OnBiometricSwitchToggled(object sender, ToggledEventArgs e)
        {
            bool isBiometricEnabled = e.Value;
            // Handle biometric toggle logic here
        }

        private async void OnMenuIconTapped(object sender, EventArgs e)
        {
            if (isSidebarVisible)
            {
                await Sidebar.TranslateTo(-250, 0, 250, Easing.CubicIn);
                Sidebar.IsVisible = false;
                Overlay.IsVisible = false;
                isSidebarVisible = false;
            }
            else
            {
                Sidebar.IsVisible = true;
                Overlay.IsVisible = true;
                await Sidebar.TranslateTo(0, 0, 250, Easing.CubicOut);
                isSidebarVisible = true;
            }
        }

        private async void OnOverlayTapped(object sender, EventArgs e) => await HideSidebarAsync();

        private async void OnApprovalClicked(object sender, EventArgs e)
        {
            await Navigation.PushAsync(new AdminPage());
            Navigation.RemovePage(this);
            await HideSidebarAsync();
        }

        private async void OnDashboardClicked(object sender, EventArgs e)
        {
            await Navigation.PushAsync(new DPage());
            Navigation.RemovePage(this);
            await HideSidebarAsync();
        }

        private async void OnImageClicked(object sender, EventArgs e)
        {
            await Navigation.PushAsync(new ImagePage());
            Navigation.RemovePage(this);
            await HideSidebarAsync();
        }

        private async System.Threading.Tasks.Task HideSidebarAsync()
        {
            if (isSidebarVisible)
            {
                await Sidebar.TranslateTo(-250, 0, 250, Easing.CubicIn);
                Sidebar.IsVisible = false;
                Overlay.IsVisible = false;
                isSidebarVisible = false;
            }
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            refreshTimer?.Stop();
            refreshTimer?.Dispose();
        }

        public class UserModel
        {
            public int Id { get; set; }
            public string Username { get; set; }
            public bool IsApproved { get; set; }

            public string Status => IsApproved ? "Approved" : "Not Approved";
            public string StatusIcon => IsApproved ? "check.png" : "wrong.png";
            public string ActionText => IsApproved ? "Disallow" : "Approve";
            public Color ActionColor => IsApproved ? Colors.Red : Colors.Green;
            public Color StatusColor => IsApproved ? Colors.Green : Colors.Red;
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

    }
}
