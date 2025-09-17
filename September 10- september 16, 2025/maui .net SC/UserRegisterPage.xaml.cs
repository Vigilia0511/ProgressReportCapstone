using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage; // For Preferences
using Plugin.Fingerprint;
using Plugin.Fingerprint.Abstractions;
using MySql.Data.MySqlClient;

#if WINDOWS
using Windows.Security.Credentials.UI;
#endif

namespace testdatabase
{
    public partial class UserRegisterPage : ContentPage
    {
        string connStr = "server=192.168.1.149;user=root;password=oneinamillion;database=smartdb;";

        public UserRegisterPage()
        {
            InitializeComponent();
            Shell.SetBackButtonBehavior(this, new BackButtonBehavior
            {
                IsVisible = false,
                IsEnabled = false
            });

            // Handle window size changes for responsive design
            this.SizeChanged += OnPageSizeChanged;
        }

        private void OnPageSizeChanged(object sender, EventArgs e)
        {
            // Use Dispatcher instead of deprecated Device.BeginInvokeOnMainThread
            Dispatcher.Dispatch(() =>
            {
                UpdateLayoutForScreenSize();
            });
        }

        private void UpdateLayoutForScreenSize()
        {
            try
            {
                var width = this.Width;

                // Return early if width is not valid yet
                if (width <= 0) return;

                // Get references to named elements with null checks
                var mainFrame = this.FindByName<Frame>("MainFrame");
                var formSection = this.FindByName<VerticalStackLayout>("FormSection");
                var logoFrame = this.FindByName<Frame>("LogoFrame");
                var welcomeText = this.FindByName<Label>("WelcomeText");
                var subtitleText = this.FindByName<Label>("SubtitleText");
                var registerButton = this.FindByName<Button>("RegisterButton");
                var infoText = this.FindByName<Label>("InfoText");
                var infoSection = this.FindByName<VerticalStackLayout>("InfoSection");

                if (mainFrame == null || formSection == null) return;

                // Desktop/Tablet layout (width >= 600)
                if (width >= 600)
                {
                    // Larger spacing and sizing for desktop
                    mainFrame.MaximumWidthRequest = 480;
                    mainFrame.MinimumWidthRequest = 400;
                    formSection.Padding = new Thickness(48);
                    formSection.Spacing = 32;

                    // Larger text
                    if (welcomeText != null) welcomeText.FontSize = 32;
                    if (subtitleText != null) subtitleText.FontSize = 18;
                    if (registerButton != null)
                    {
                        registerButton.HeightRequest = 56;
                        registerButton.FontSize = 18;
                    }

                    // More detailed info
                    if (infoText != null)
                        infoText.Text = "Secure biometric authentication required";

                    // Vertical layout for info on desktop for better readability
                    if (infoSection != null) infoSection.Spacing = 20;
                }
                // Mobile layout (width < 600)
                else
                {
                    // Compact sizing for mobile
                    mainFrame.MaximumWidthRequest = 420;
                    mainFrame.MinimumWidthRequest = 320;
                    formSection.Padding = new Thickness(32);
                    formSection.Spacing = 24;

                    // Standard text sizes
                    if (welcomeText != null) welcomeText.FontSize = 28;
                    if (subtitleText != null) subtitleText.FontSize = 16;
                    if (registerButton != null)
                    {
                        registerButton.HeightRequest = 48;
                        registerButton.FontSize = 16;
                    }

                    // Shorter info text
                    if (infoText != null)
                        infoText.Text = "Biometric authentication required";

                    // Compact spacing for mobile
                    if (infoSection != null) infoSection.Spacing = 16;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error updating layout: {ex.Message}");
            }
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            // Use Dispatcher instead of deprecated Device.BeginInvokeOnMainThread
            Dispatcher.Dispatch(() =>
            {
                UpdateLayoutForScreenSize();
            });
        }

        private async void OnBiometricButtonClicked(object sender, EventArgs e)
        {
            try
            {
                // Add subtle visual feedback
                var button = sender as Button;
                if (button != null)
                {
                    // Gentle scale animation
                    await button.ScaleTo(0.98, 80);
                    await button.ScaleTo(1.0, 80);
                }

                bool authenticationSuccessful = false;

#if WINDOWS
                // Use Windows Hello for Windows platform
                authenticationSuccessful = await AuthenticateWithWindowsHello();
                
                if (!authenticationSuccessful)
                {
                    await DisplayAlert("Windows Hello Failed",
                        "Windows Hello authentication failed or is not available. Please ensure Windows Hello is set up on your device.",
                        "OK");
                    return;
                }
#else
                // Use Plugin.Fingerprint for other platforms (Android, iOS)
                var isAvailable = await CrossFingerprint.Current.IsAvailableAsync(true);

                if (!isAvailable)
                {
                    await DisplayAlert("Biometric Unavailable",
                        "Biometric authentication is not available on this device. Please ensure your device supports fingerprint or face recognition.",
                        "OK");
                    return;
                }

                var result = await CrossFingerprint.Current.AuthenticateAsync(
                    new AuthenticationRequestConfiguration(
                        "Biometric Registration",
                        "Scan your fingerprint or use face recognition to register your account securely")
                    {
                        CancelTitle = "Cancel",
                        FallbackTitle = "Use PIN",
                        AllowAlternativeAuthentication = true
                    });

                if (result.Authenticated)
                {
                    authenticationSuccessful = true;
                }
                else
                {
                    string errorMessage = result.ErrorMessage ?? "Authentication failed";
                    await DisplayAlert("Authentication Failed",
                        $"Biometric authentication was not successful: {errorMessage}",
                        "Try Again");
                    return;
                }
#endif

                if (authenticationSuccessful)
                {
                    await OnBiometricSuccess("RecognizedUser");
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error",
                    $"An error occurred during authentication: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"Authentication error: {ex}");
            }
        }

#if WINDOWS
        private async Task<bool> AuthenticateWithWindowsHello()
        {
            try
            {
                // Check if Windows Hello is available
                var availability = await UserConsentVerifier.CheckAvailabilityAsync();
                
                if (availability == UserConsentVerifierAvailability.Available)
                {
                    // Request Windows Hello authentication
                    var result = await UserConsentVerifier.RequestVerificationAsync("Please verify your identity to register your account securely");
                    return result == UserConsentVerificationResult.Verified;
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"Windows Hello not available: {availability}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Windows Hello authentication error: {ex}");
                return false;
            }
        }
#endif

        private async Task OnBiometricSuccess(string recognizedName)
        {
            string fullname = await DisplayPromptAsync(
                "Complete Registration",
                "Please enter your full name to complete the registration process:",
                "Register",
                "Cancel",
                "Enter your full name...",
                maxLength: 50);

            if (!string.IsNullOrWhiteSpace(fullname))
            {
                // Trim and validate the name
                fullname = fullname.Trim();

                if (fullname.Length < 2)
                {
                    await DisplayAlert("Invalid Name", "Please enter a valid full name (at least 2 characters).", "OK");
                    return;
                }

                bool isNameExists = await IsNameExistsInDatabase(fullname);

                if (!isNameExists)
                {
                    await SaveFullNameToDatabase(fullname);

                    // Store full name in Preferences for login use
                    Preferences.Set("RegisteredFullName", fullname);
                }
                else
                {
                    await DisplayAlert("Name Already Exists",
                        "This name is already registered in our system. Please choose a different name or contact support if this is your account.",
                        "OK");
                }
            }
            else
            {
                await DisplayAlert("Registration Cancelled", "Full name is required to complete registration.", "OK");
            }
        }

        private async Task<bool> IsNameExistsInDatabase(string fullname)
        {
            try
            {
                using MySqlConnection conn = new(connStr);
                await conn.OpenAsync();

                string checkQuery = "SELECT COUNT(*) FROM users WHERE LOWER(username) = LOWER(@username)";
                using MySqlCommand cmd = new(checkQuery, conn);
                cmd.Parameters.AddWithValue("@username", fullname);

                var result = await cmd.ExecuteScalarAsync();
                int count = Convert.ToInt32(result);

                return count > 0;
            }
            catch (MySqlException ex)
            {
                await DisplayAlert("Database Connection Error",
                    $"Unable to connect to the database: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"Database error: {ex}");
                return true; // Assume exists to prevent duplicate registration
            }
            catch (Exception ex)
            {
                await DisplayAlert("Database Error",
                    $"Unable to verify username availability: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"General database error: {ex}");
                return true; // Assume exists to prevent duplicate registration
            }
        }

        private async Task SaveFullNameToDatabase(string fullname)
        {
            try
            {
                using MySqlConnection conn = new(connStr);
                await conn.OpenAsync();

                string insertQuery = @"INSERT INTO users (username, is_approved, timestamp) 
                                     VALUES (@username, @is_approved, @timestamp)";
                using MySqlCommand cmd = new(insertQuery, conn);
                cmd.Parameters.AddWithValue("@username", fullname);
                cmd.Parameters.AddWithValue("@is_approved", false);
                cmd.Parameters.AddWithValue("@timestamp", DateTime.Now);

                await cmd.ExecuteNonQueryAsync();

                await DisplayAlert("Registration Successful! 🎉",
                    "Your account has been created successfully! Please wait for admin approval before you can log in. You will be notified once your account is activated.",
                    "Continue");
            }
            catch (MySqlException ex)
            {
                await DisplayAlert("Database Connection Error",
                    $"Unable to connect to the database: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"Database error: {ex}");
            }
            catch (Exception ex)
            {
                await DisplayAlert("Registration Error",
                    $"Unable to complete registration: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"General registration error: {ex}");
            }
        }

        private async void OnLoginTapped(object sender, EventArgs e)
        {
            try
            {
                // Add subtle visual feedback for tap
                var label = sender as Label;
                if (label != null)
                {
                    var originalColor = label.TextColor;
                    label.TextColor = Color.FromArgb("#4C51BF"); // Slightly darker blue
                    await Task.Delay(80);
                    label.TextColor = originalColor;
                }

                await Navigation.PushAsync(new UserLoginPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error",
                    $"Unable to navigate to login page: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"Navigation error: {ex}");
            }
        }
    }
}