using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage; // For Preferences
using Plugin.Fingerprint;
using Plugin.Fingerprint.Abstractions;
using MySql.Data.MySqlClient;
using System;

#if WINDOWS
using Windows.Security.Credentials.UI;
#endif

namespace testdatabase
{
    public partial class UserLoginPage : ContentPage
    {
        string connStr = "server=192.168.1.149;user=root;password=oneinamillion;database=smartdb;";

        public UserLoginPage()
        {
            InitializeComponent();
            Shell.SetBackButtonBehavior(this, new BackButtonBehavior
            {
                IsVisible = false,
                IsEnabled = false
            });

            // Handle window size changes for responsive design
            this.SizeChanged += OnPageSizeChanged;

            // Display the registered name if it exists
            LoadRegisteredUserName();
        }

        private void LoadRegisteredUserName()
        {
            try
            {
                var registeredName = Preferences.Get("RegisteredFullName", string.Empty);
                if (NameLabel != null)
                {
                    NameLabel.Text = !string.IsNullOrWhiteSpace(registeredName) ? registeredName : "No user registered";
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading registered name: {ex.Message}");
            }
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
                var loginButton = this.FindByName<Button>("LoginButton");
                var infoText = this.FindByName<Label>("InfoText");

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
                    if (loginButton != null)
                    {
                        loginButton.HeightRequest = 56;
                        loginButton.FontSize = 18;
                    }

                    // More detailed info
                    if (infoText != null)
                        infoText.Text = "Use your fingerprint or face recognition to sign in securely";
                }
                // Mobile layout (width < 600)
                else
                {
                    // Compact sizing for mobile
                    mainFrame.MaximumWidthRequest = 400;
                    mainFrame.MinimumWidthRequest = 320;
                    formSection.Padding = new Thickness(32);
                    formSection.Spacing = 24;

                    // Standard text sizes
                    if (welcomeText != null) welcomeText.FontSize = 28;
                    if (subtitleText != null) subtitleText.FontSize = 16;
                    if (loginButton != null)
                    {
                        loginButton.HeightRequest = 48;
                        loginButton.FontSize = 16;
                    }

                    // Shorter info text
                    if (infoText != null)
                        infoText.Text = "Use biometric authentication to sign in";
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
                LoadRegisteredUserName();
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
                    await button.ScaleTo(0.98, 80);
                    await button.ScaleTo(1.0, 80);
                }

                bool authenticationSuccessful = false;

#if WINDOWS
                // Use Windows Hello for Windows platform
                authenticationSuccessful = await AuthenticateWithWindowsHello();
                
                // If Windows Hello fails or is not available, fall back to password
                if (!authenticationSuccessful)
                {
                    var useWindowsAuth = await DisplayAlert("Windows Hello Failed",
                        "Windows Hello authentication failed or is not available. Would you like to use your Windows password instead?",
                        "Yes", "Cancel");

                    if (useWindowsAuth)
                    {
                        authenticationSuccessful = await AuthenticateWithWindowsCredentials();
                    }
                }
#else
                // Use Plugin.Fingerprint for other platforms (Android, iOS)
                var isAvailable = await CrossFingerprint.Current.IsAvailableAsync(true);

                if (isAvailable)
                {
                    // Use biometric authentication
                    var authRequest = new AuthenticationRequestConfiguration(
                        "Biometric Login",
                        "Scan your fingerprint or use face recognition to sign in securely")
                    {
                        CancelTitle = "Cancel",
                        FallbackTitle = "Use Password",
                        AllowAlternativeAuthentication = true
                    };

                    var result = await CrossFingerprint.Current.AuthenticateAsync(authRequest);
                    authenticationSuccessful = result.Authenticated;

                    if (!result.Authenticated && !string.IsNullOrEmpty(result.ErrorMessage))
                    {
                        await DisplayAlert("Authentication Failed", result.ErrorMessage, "OK");
                    }
                }
                else
                {
                    await DisplayAlert("Biometric Not Available",
                        "Biometric authentication is not available on this device.", "OK");
                }
#endif

                if (authenticationSuccessful)
                {
                    await ProcessSuccessfulAuthentication();
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
                    var result = await UserConsentVerifier.RequestVerificationAsync("Please verify your identity to sign in to the application");
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

        private async Task<bool> AuthenticateWithWindowsCredentials()
        {
#if WINDOWS
            try
            {
                // Show a prompt asking for Windows password
                string password = await DisplayPromptAsync("Windows Authentication",
                    $"Enter your Windows password for user: {System.Environment.UserName}",
                    "OK", "Cancel",
                    placeholder: "Windows Password",
                    keyboard: Keyboard.Default);

                if (string.IsNullOrWhiteSpace(password))
                {
                    return false; // User cancelled or didn't enter password
                }

                // Get current Windows username and domain
                string currentUser = System.Environment.UserName;
                string domain = System.Environment.UserDomainName;

                // Attempt to validate Windows credentials using P/Invoke
                return await Task.Run(() => ValidateWindowsCredentials(domain, currentUser, password));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Windows authentication error: {ex}");
                await DisplayAlert("Authentication Error",
                    "Failed to authenticate with Windows credentials.", "OK");
                return false;
            }
#else
            await DisplayAlert("Not Supported",
                "Windows authentication is only available on Windows platform.", "OK");
            return false;
#endif
        }

#if WINDOWS
        [System.Runtime.InteropServices.DllImport("advapi32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        private static extern bool LogonUser(
            string lpszUsername,
            string lpszDomain,
            string lpszPassword,
            int dwLogonType,
            int dwLogonProvider,
            out IntPtr phToken);

        [System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto)]
        private static extern bool CloseHandle(IntPtr handle);

        private bool ValidateWindowsCredentials(string domain, string username, string password)
        {
            IntPtr token = IntPtr.Zero;
            try
            {
                // Try different logon types to handle various Windows authentication scenarios

                // First try: Interactive logon (standard desktop login)
                bool result = LogonUser(username, domain, password, 2, 0, out token);
                if (result && token != IntPtr.Zero)
                {
                    CloseHandle(token);
                    return true;
                }
                if (token != IntPtr.Zero) CloseHandle(token);

                // Second try: Network logon (for domain accounts)
                result = LogonUser(username, domain, password, 3, 0, out token);
                if (result && token != IntPtr.Zero)
                {
                    CloseHandle(token);
                    return true;
                }
                if (token != IntPtr.Zero) CloseHandle(token);

                // Third try: Local machine account (try with computer name as domain)
                string computerName = System.Environment.MachineName;
                result = LogonUser(username, computerName, password, 2, 0, out token);
                if (result && token != IntPtr.Zero)
                {
                    CloseHandle(token);
                    return true;
                }
                if (token != IntPtr.Zero) CloseHandle(token);

                // Fourth try: Try with empty domain for local accounts
                result = LogonUser(username, "", password, 2, 0, out token);
                if (result && token != IntPtr.Zero)
                {
                    CloseHandle(token);
                    return true;
                }
                if (token != IntPtr.Zero) CloseHandle(token);

                // Log the last error for debugging
                int lastError = System.Runtime.InteropServices.Marshal.GetLastWin32Error();
                System.Diagnostics.Debug.WriteLine($"LogonUser failed with error code: {lastError}");

                return false;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Windows credential validation error: {ex}");
                if (token != IntPtr.Zero)
                {
                    CloseHandle(token);
                }
                return false;
            }
        }
#endif

        private async Task ProcessSuccessfulAuthentication()
        {
            // Retrieve registered username
            var registeredName = Preferences.Get("RegisteredFullName", string.Empty);
            if (string.IsNullOrWhiteSpace(registeredName))
            {
                await DisplayAlert("No User Found",
                    "No registered user found. Please register first.",
                    "OK");
                return;
            }

            // Check approval
            bool isApproved = await CheckIfUserIsApproved(registeredName);
            if (isApproved)
            {
                // ✅ Store user name for sidebar use
                Preferences.Set("LoggedInUser", registeredName);

                await Navigation.PushAsync(new DPage());
                Navigation.RemovePage(this);
            }
            else
            {
                await DisplayAlert("Pending Approval",
                    "Your account is still pending admin approval. Please wait for activation before signing in.",
                    "OK");
            }
        }


        private async Task<bool> CheckIfUserIsApproved(string fullname)
        {
            try
            {
                using var conn = new MySqlConnection(connStr);
                await conn.OpenAsync();

                string query = "SELECT is_approved FROM users WHERE LOWER(username) = LOWER(@username)";
                using var cmd = new MySqlCommand(query, conn);
                cmd.Parameters.AddWithValue("@username", fullname);

                var result = await cmd.ExecuteScalarAsync();
                return result != null && Convert.ToBoolean(result);
            }
            catch (MySqlException ex)
            {
                await DisplayAlert("Database Connection Error",
                    $"Unable to connect to the database: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"Database error: {ex}");
                return false;
            }
            catch (Exception ex)
            {
                await DisplayAlert("Database Error",
                    $"Unable to verify account status: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"General database error: {ex}");
                return false;
            }
        }

        private async void OnRegisterTapped(object sender, EventArgs e)
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

                await Navigation.PushAsync(new UserRegisterPage());
                Navigation.RemovePage(this);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Navigation Error",
                    $"Unable to navigate to registration page: {ex.Message}",
                    "OK");
                System.Diagnostics.Debug.WriteLine($"Navigation error: {ex}");
            }
        }
    }
}