using MySql.Data.MySqlClient;
using System.Security.Cryptography;
using System.Text;
using Plugin.Fingerprint;
using Plugin.Fingerprint.Abstractions;
#if WINDOWS
using Windows.Security.Credentials.UI;
#endif

namespace testdatabase;

public partial class AdminPagelogin : ContentPage
{
    private int _loginAttempts = 0;
    private const int MAX_LOGIN_ATTEMPTS = 3;
    private DateTime _lockoutEndTime = DateTime.MinValue;

    public AdminPagelogin()
    {
        InitializeComponent();
        SetupPage();
        SetupEntryEvents();

        // Subscribe to biometric settings changes
        MessagingCenter.Subscribe<object>(this, "BiometricSettingsChanged", async (sender) =>
        {
            await SetupBiometricUI();
        });
    }

    // Add this method to be called when the page is disposed
    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        MessagingCenter.Unsubscribe<object>(this, "BiometricSettingsChanged");
    }

    // Public method to trigger biometric auth externally (from another page)
    public async Task<bool> TriggerBiometricAuthenticationAsync()
    {
        return await AuthenticateAsync("Verify your identity to sign in");
    }

    // Button click handler that triggers authentication
    private async void OnBiometricButtonClicked(object sender, EventArgs e)
    {
        var button = sender as Button;
        if (button != null)
        {
            await button.ScaleTo(0.98, 80);
            await button.ScaleTo(1.0, 80);
        }

        bool authSuccess = await AuthenticateAsync("Verify your identity to sign in");

        if (authSuccess)
        {
            await ProcessSuccessfulAuthentication();
        }
        else
        {
            await DisplayAlert("Authentication Failed", "Biometric authentication failed or cancelled.", "OK");
        }
    }

    private async Task ProcessSuccessfulAuthentication()
    {
        // Reset login attempts on successful biometric auth
        _loginAttempts = 0;

        await DisplayAlert("Welcome!", "Biometric authentication successful. Redirecting to admin panel...", "Continue");

        // Smooth transition
        await this.FadeTo(0, 300);
        await Navigation.PushAsync(new AdminPage());
        Navigation.RemovePage(this);
    }

    // Biometric authentication methods
    private async Task<bool> AuthenticateAsync(string reason = "Authenticate to continue")
    {
#if WINDOWS
        return await AuthenticateWithWindowsFlowAsync(reason);
#else
        return await AuthenticateWithFingerprintAsync(reason);
#endif
    }

    public async Task<bool> IsBiometricAvailableAsync()
    {
#if WINDOWS
        try
        {
            var availability = await UserConsentVerifier.CheckAvailabilityAsync();
            return availability == UserConsentVerifierAvailability.Available;
        }
        catch
        {
            return false;
        }
#else
        try
        {
            return await CrossFingerprint.Current.IsAvailableAsync(true);
        }
        catch
        {
            return false;
        }
#endif
    }

    private async Task<bool> AuthenticateWithFingerprintAsync(string reason)
    {
        try
        {
            var isAvailable = await CrossFingerprint.Current.IsAvailableAsync(true);
            if (!isAvailable)
            {
                return false;
            }

            var config = new AuthenticationRequestConfiguration("Biometric Authentication", reason)
            {
                CancelTitle = "Cancel",
                FallbackTitle = "Use Password",
                AllowAlternativeAuthentication = true,
            };

            var result = await CrossFingerprint.Current.AuthenticateAsync(config);
            return result.Authenticated;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Biometric authentication error: {ex.Message}");
            return false;
        }
    }

#if WINDOWS
    private async Task<bool> AuthenticateWithWindowsFlowAsync(string reason)
    {
        return await AuthenticateWithWindowsHello(reason);
    }

    private async Task<bool> AuthenticateWithWindowsHello(string reason)
    {
        try
        {
            var availability = await UserConsentVerifier.CheckAvailabilityAsync();
            if (availability == UserConsentVerifierAvailability.Available)
            {
                var result = await UserConsentVerifier.RequestVerificationAsync(reason);
                return result == UserConsentVerificationResult.Verified;
            }
            return false;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Windows Hello authentication error: {ex.Message}");
            return false;
        }
    }
#endif

    private void SetupPage()
    {
        Shell.SetBackButtonBehavior(this, new BackButtonBehavior
        {
            IsVisible = false,
            IsEnabled = false
        });

        // Add subtle entrance animation
        this.Opacity = 0;
        this.FadeTo(1, 500, Easing.CubicOut);
    }

    private void SetupEntryEvents()
    {
        // Add Enter key support and focus management
        usernameEntry.Completed += (s, e) => passwordEntry.Focus();
        passwordEntry.Completed += async (s, e) => await HandleLogin();

        // Add visual feedback on focus
        usernameEntry.Focused += OnEntryFocused;
        usernameEntry.Unfocused += OnEntryUnfocused;
        passwordEntry.Focused += OnEntryFocused;
        passwordEntry.Unfocused += OnEntryUnfocused;
    }

    private void OnEntryFocused(object sender, FocusEventArgs e)
    {
        if (sender is Entry entry && entry.Parent is Frame frame)
        {
            frame.BorderColor = Color.FromArgb("#667eea");
            frame.BackgroundColor = Color.FromArgb("#ffffff");
        }
    }

    private void OnEntryUnfocused(object sender, FocusEventArgs e)
    {
        if (sender is Entry entry && entry.Parent is Frame frame)
        {
            frame.BorderColor = Color.FromArgb("#e2e8f0");
            frame.BackgroundColor = Color.FromArgb("#f8fafc");
        }
    }

    // Updated method - now public so it can be called from messaging
    public async Task SetupBiometricUI()
    {
        try
        {
            // Check if biometric is enabled in preferences
            bool isBiometricEnabled = Preferences.Get("BiometricEnabled", false);

            System.Diagnostics.Debug.WriteLine($"Biometric enabled in preferences: {isBiometricEnabled}");

            if (isBiometricEnabled)
            {
                // Check if biometric is actually available on device
                bool isAvailable = await IsBiometricAvailableAsync();

                System.Diagnostics.Debug.WriteLine($"Biometric available on device: {isAvailable}");

                if (isAvailable)
                {
                    // Show the biometric section with animation
                    biometricSection.IsVisible = true;
                    biometricSection.Opacity = 0;
                    await biometricSection.FadeTo(1, 300);
                    System.Diagnostics.Debug.WriteLine("Biometric section made visible");
                }
                else
                {
                    // Biometric was enabled but is no longer available
                    Preferences.Set("BiometricEnabled", false);
                    await HideBiometricSection();
                    System.Diagnostics.Debug.WriteLine("Biometric not available, disabled in preferences");
                }
            }
            else
            {
                // Biometric is disabled, hide the section
                await HideBiometricSection();
                System.Diagnostics.Debug.WriteLine("Biometric disabled in preferences");
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error in SetupBiometricUI: {ex.Message}");
            await HideBiometricSection();
        }
    }

    private async Task HideBiometricSection()
    {
        if (biometricSection.IsVisible)
        {
            await biometricSection.FadeTo(0, 300);
            biometricSection.IsVisible = false;
        }
    }

    // Update your existing OnAppearing method to include the biometric setup
    protected override async void OnAppearing()
    {
        base.OnAppearing();
        NavigationPage.SetHasBackButton(this, false);
        NavigationPage.SetHasNavigationBar(this, false);

        // Setup biometric UI based on user preferences
        await SetupBiometricUI();

        // Focus on username field for better UX
        usernameEntry.Focus();
    }

    // Or if you prefer to do it in the constructor/InitializeComponent area:
    private void LoadSettings()
    {
        // Load the saved biometric preference - this is handled in SetupBiometricUI()
        // No UI switch to set here since this is the login page, not settings page
        bool isBiometricEnabled = Preferences.Get("BiometricEnabled", false);
        System.Diagnostics.Debug.WriteLine($"Biometric setting loaded: {isBiometricEnabled}");
    }

    // Optional: Add this method to handle auto-prompt if biometric is enabled
    private async Task CheckForAutoBiometric()
    {
        bool isBiometricEnabled = Preferences.Get("BiometricEnabled", false);
        bool autoPrompt = Preferences.Get("BiometricAutoPrompt", false); // You can add this setting too

        if (isBiometricEnabled && autoPrompt && await IsBiometricAvailableAsync())
        {
            // Auto-prompt for biometric authentication
            bool authSuccess = await AuthenticateAsync("Verify your identity to sign in");
            if (authSuccess)
            {
                await ProcessSuccessfulAuthentication();
            }
        }
    }

    private async void OnLoginClicked(object sender, EventArgs e)
    {
        await HandleLogin();
    }

    private async Task HandleLogin()
    {
        // Check if account is locked out
        if (DateTime.Now < _lockoutEndTime)
        {
            var remainingTime = (_lockoutEndTime - DateTime.Now).Minutes + 1;
            await ShowErrorAlert($"Account temporarily locked. Try again in {remainingTime} minute(s).", "Account Locked");
            return;
        }

        string username = usernameEntry.Text?.Trim();
        string password = passwordEntry.Text?.Trim();

        // Input validation with better UX
        if (string.IsNullOrEmpty(username))
        {
            await ShowErrorAlert("Please enter your admin ID.", "Missing ID");
            usernameEntry.Focus();
            return;
        }

        if (string.IsNullOrEmpty(password))
        {
            await ShowErrorAlert("Please enter your password.", "Missing Password");
            passwordEntry.Focus();
            return;
        }

        // Show loading state
        await ShowLoadingState(true);

        try
        {
            // Simulate network delay for better UX perception
            await Task.Delay(800);

            // Admin authentication
            if (await AuthenticateAdmin(username, password))
            {
                await ShowSuccessAndNavigate();
            }
            else
            {
                await HandleFailedLogin();
            }
        }
        catch (Exception ex)
        {
            await ShowErrorAlert("An error occurred during login. Please try again.", "Login Error");
            System.Diagnostics.Debug.WriteLine($"Login error: {ex.Message}");
        }
        finally
        {
            await ShowLoadingState(false);
        }
    }

    private async Task<bool> AuthenticateAdmin(string username, string password)
    {
        // Primary admin bypass
        if (username == "051123" && password == "admin11")
        {
            return true;
        }

        // Add more admin accounts if needed
        var adminAccounts = new Dictionary<string, string>
        {
            { "051123", "admin11" },
            // Add more admin accounts here
        };

        return adminAccounts.ContainsKey(username) && adminAccounts[username] == password;
    }

    private async Task ShowSuccessAndNavigate()
    {
        _loginAttempts = 0; // Reset attempts on success

        // Success feedback
        await loginButton.ScaleTo(0.95, 100);
        await loginButton.ScaleTo(1.0, 100);

        await DisplayAlert("Welcome!", "Login successful. Redirecting to admin panel...", "Continue");

        // Smooth transition
        await this.FadeTo(0, 300);
        await Navigation.PushAsync(new AdminPage());
        Navigation.RemovePage(this);
    }

    private async Task HandleFailedLogin()
    {
        _loginAttempts++;

        // Visual feedback for failed attempt - change button color temporarily
        var originalBackground = loginButton.Background;
        loginButton.BackgroundColor = Color.FromArgb("#ef4444");
        await Task.Delay(300);

        // Restore original gradient background
        loginButton.BackgroundColor = Colors.Transparent;
        loginButton.Background = new LinearGradientBrush
        {
            StartPoint = new Point(0, 0),
            EndPoint = new Point(1, 0),
            GradientStops = new GradientStopCollection
            {
                new GradientStop { Color = Color.FromArgb("#667eea"), Offset = 0.0f },
                new GradientStop { Color = Color.FromArgb("#764ba2"), Offset = 1.0f }
            }
        };

        // Shake animation for failed login
        await ShakeAnimation();

        if (_loginAttempts >= MAX_LOGIN_ATTEMPTS)
        {
            _lockoutEndTime = DateTime.Now.AddMinutes(5);
            await ShowErrorAlert($"Too many failed attempts. Account locked for 5 minutes.", "Account Locked");
        }
        else
        {
            var remainingAttempts = MAX_LOGIN_ATTEMPTS - _loginAttempts;
            await ShowErrorAlert($"Invalid credentials. {remainingAttempts} attempt(s) remaining.", "Login Failed");
        }

        // Clear password field for security
        passwordEntry.Text = string.Empty;
        usernameEntry.Focus();
    }

    private async Task ShakeAnimation()
    {
        var frame = (Frame)loginButton.Parent.Parent;
        await frame.TranslateTo(-10, 0, 50);
        await frame.TranslateTo(10, 0, 50);
        await frame.TranslateTo(-5, 0, 50);
        await frame.TranslateTo(5, 0, 50);
        await frame.TranslateTo(0, 0, 50);
    }

    private async Task ShowLoadingState(bool isLoading)
    {
        loginButton.IsEnabled = !isLoading;
        usernameEntry.IsEnabled = !isLoading;
        passwordEntry.IsEnabled = !isLoading;

        if (isLoading)
        {
            loginButton.Text = "Signing In...";
            loginButton.Opacity = 0.7;
            loadingIndicator.IsVisible = true;
            loadingIndicator.IsRunning = true;
        }
        else
        {
            loginButton.Text = "Sign In";
            loginButton.Opacity = 1.0;
            loadingIndicator.IsVisible = false;
            loadingIndicator.IsRunning = false;
        }
    }

    private async Task ShowErrorAlert(string message, string title)
    {
        await DisplayAlert(title, message, "OK");
    }
}