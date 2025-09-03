using Microsoft.Maui.Controls;

namespace testdatabase
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();

            // Hide navigation bar and back button
            Shell.SetBackButtonBehavior(this, new BackButtonBehavior
            {
                IsVisible = false,
                IsEnabled = false
            });

            // Add subtle entrance animation
            this.Opacity = 0;
            this.FadeTo(1, 600, Easing.CubicOut);
        }

        private async void OnUserButtonClicked(object sender, EventArgs e)
        {
            var button = sender as Button;

            // Button press animation
            await AnimateButtonPress(button);

            // Navigate with smooth transition
            await Navigation.PushAsync(new UserLoginPage());
            Navigation.RemovePage(this);
        }

        private async void OnAdminButtonClicked(object sender, EventArgs e)
        {
            var button = sender as Button;

            // Button press animation
            await AnimateButtonPress(button);

            // Navigate with smooth transition
            await Navigation.PushAsync(new AdminPagelogin());
            Navigation.RemovePage(this);
        }

        private async Task AnimateButtonPress(Button button)
        {
            if (button == null) return;

            // Scale down and back up for press effect
            await Task.WhenAll(
                button.ScaleTo(0.95, 100, Easing.CubicOut),
                button.FadeTo(0.8, 100, Easing.CubicOut)
            );

            await Task.WhenAll(
                button.ScaleTo(1.0, 100, Easing.CubicOut),
                button.FadeTo(1.0, 100, Easing.CubicOut)
            );
        }
    }
}