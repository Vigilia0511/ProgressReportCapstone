public static class BiometricHelper
{
    public static async Task<bool> IsAvailableAsync()
    {
#if WINDOWS
        try
        {
            var availability = await Windows.Security.Credentials.UI.UserConsentVerifier.CheckAvailabilityAsync();
            return availability == Windows.Security.Credentials.UI.UserConsentVerifierAvailability.Available;
        }
        catch { return false; }
#else
        try
        {
            return await Plugin.Fingerprint.CrossFingerprint.Current.IsAvailableAsync(true);
        }
        catch { return false; }
#endif
    }

    public static async Task<bool> TriggerBiometricAuthenticationAsync(string reason = "Verify your identity to sign in")
    {
#if WINDOWS
        try
        {
            var result = await Windows.Security.Credentials.UI.UserConsentVerifier.RequestVerificationAsync(reason);
            return result == Windows.Security.Credentials.UI.UserConsentVerificationResult.Verified;
        }
        catch { return false; }
#else
        try
        {
            var config = new Plugin.Fingerprint.Abstractions.AuthenticationRequestConfiguration("Biometric Authentication", reason)
            {
                CancelTitle = "Cancel",
                FallbackTitle = "Use Password",
                AllowAlternativeAuthentication = true,
            };
            var result = await Plugin.Fingerprint.CrossFingerprint.Current.AuthenticateAsync(config);
            return result.Authenticated;
        }
        catch { return false; }
#endif
    }
}
