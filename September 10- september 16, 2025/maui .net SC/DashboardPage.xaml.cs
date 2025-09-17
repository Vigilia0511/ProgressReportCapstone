namespace testdatabase;

public partial class DashboardPage : ContentPage
{
    public DashboardPage()
    {
        InitializeComponent();

    }

    private void OnFaceClicked(object sender, EventArgs e)
    {
        // Navigate to Face Authentication Page or perform face auth logic
        DisplayAlert("Face", "Face authentication selected.", "OK");
    }

    private void OnFingerprintClicked(object sender, EventArgs e)
    {
        // Navigate to Fingerprint Authentication Page or perform fingerprint auth logic
        DisplayAlert("Fingerprint", "Fingerprint authentication selected.", "OK");
    }

    private void OnPinClicked(object sender, EventArgs e)
    {
        // Navigate to PIN entry page or perform PIN auth logic
        DisplayAlert("PIN", "PIN authentication selected.", "OK");
    }

    private void OnVoiceClicked(object sender, EventArgs e)
    {
        // Navigate to Voice Authentication Page or perform voice auth logic
        DisplayAlert("Voice", "Voice authentication selected.", "OK");
    }
}
