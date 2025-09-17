using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;

namespace testdatabase.Helpers
{
    public static class ThemeManager
    {
        public static bool IsDarkModeEnabled => Preferences.Get("DarkMode", false);

        public static void ApplyTheme(Page page)
        {
            if (IsDarkModeEnabled)
                ApplyDarkTheme(page);
            else
                ApplyLightTheme(page);
        }

        public static void ApplyDarkTheme(Page page)
        {
            page.BackgroundColor = Color.FromArgb("#000000"); // Black

            if (page.FindByName<StackLayout>("SettingsDropdown") is StackLayout dropdown)
                dropdown.BackgroundColor = Color.FromArgb("#2C2C2C");

            if (page.FindByName<Label>("NotificationLabel") is Label notificationLabel)
                notificationLabel.TextColor = Colors.White;

            if (page.FindByName<Label>("TitleLabel") is Label titleLabel)
                titleLabel.TextColor = Colors.White;

            if (page.FindByName<Label>("NameLabel") is Label nameLabel)
                nameLabel.TextColor = Colors.White;

            if (page.FindByName<Label>("UsernameLabel") is Label usernameLabel)
                usernameLabel.TextColor = Colors.White;

            if (page.FindByName<Label>("CapturedImagesLabel") is Label capturedImagesLabel)
                capturedImagesLabel.TextColor = Colors.White; // for dark mode

            Application.Current.Resources["UsernameTextColorLight"] = Colors.White;

            if (page.FindByName<Label>("NameLabel1") is Label nameLabel1)
                nameLabel1.TextColor = Colors.Black;



        }

        public static void ApplyLightTheme(Page page)
        {
            page.BackgroundColor = Color.FromArgb("#FFFFFF"); // White

            if (page.FindByName<StackLayout>("SettingsDropdown") is StackLayout dropdown)
                dropdown.BackgroundColor = Color.FromArgb("#3B84B1");

            if (page.FindByName<Label>("NotificationLabel") is Label notificationLabel)
                notificationLabel.TextColor = Colors.Black;

            if (page.FindByName<Label>("TitleLabel") is Label titleLabel)
                titleLabel.TextColor = Colors.Black;

            if (page.FindByName<Label>("NameLabel") is Label nameLabel)
                nameLabel.TextColor = Colors.Black;

            if (page.FindByName<Label>("UsernameLabel") is Label usernameLabel)
                usernameLabel.TextColor = Colors.Black;

            if (page.FindByName<Label>("CapturedImagesLabel") is Label capturedImagesLabel)
                capturedImagesLabel.TextColor = Color.FromArgb("#002B45"); // for light mode

            Application.Current.Resources["UsernameTextColorLight"] = Colors.Black;

            if (page.FindByName<Label>("NameLabel1") is Label nameLabel1)
                nameLabel1.TextColor = Colors.Black;


        }



        public static void SetDarkMode(bool enabled, Page page)
        {
            Preferences.Set("DarkMode", enabled);

            if (enabled)
                ApplyDarkTheme(page);
            else
                ApplyLightTheme(page);
        }
    }
}
