using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;
using MySql.Data.MySqlClient;
using Microsoft.Maui.ApplicationModel; // For MainThread
using Microsoft.Maui.Storage; // For Preferences
using testdatabase.Helpers;  // <-- Add this to access ThemeManager

namespace testdatabase
{
    public partial class ImagePage : ContentPage
    {
        bool isSidebarVisible = false;
        private readonly string _connStr = "server=192.168.1.149;user=root;password=oneinamillion;database=smartdb;";
        private System.Timers.Timer _refreshTimer;
        private DateTime _lastImageTimestamp = DateTime.MinValue;
        private int _lastImageCount = 0;
        private bool _isPageActive = true;
        private HashSet<int> _loadedImageIds = new HashSet<int>(); // Track loaded images
        private bool _isLoadingImages = false; // Prevent concurrent loading
        private readonly object _loadingLock = new object(); // Thread safety
        private string _loggedInUser = string.Empty;
        private bool isSettingsExpanded = false;
        private bool _isLoadingSettings = false; // Add this flag to prevent event triggers during loading

        public ImagePage()
        {
            InitializeComponent();

            LoadSettings();

            // Get logged in user
            _loggedInUser = Preferences.Get("LoggedInUser", string.Empty);

            // Update sidebar with username
            UpdateSidebarUsername();

            _ = LoadCapturedImages(); // Fire and forget for constructor
            Shell.SetBackButtonBehavior(this, new BackButtonBehavior
            {
                IsVisible = false,
                IsEnabled = false
            });
            StartAutoRefresh();
        }

        private async void OnHamburgerClicked(object sender, EventArgs e)
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

        private async void OnOverlayTapped(object sender, EventArgs e)
        {
            if (isSidebarVisible)
            {
                await Sidebar.TranslateTo(-250, 0, 250, Easing.CubicIn);
                Sidebar.IsVisible = false;
                Overlay.IsVisible = false;
                isSidebarVisible = false;
            }
        }

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

        private async Task HideSidebarAsync()
        {
            if (isSidebarVisible)
            {
                await Sidebar.TranslateTo(-250, 0, 250, Easing.CubicIn);
                Sidebar.IsVisible = false;
                Overlay.IsVisible = false;
                isSidebarVisible = false;
            }
        }

        private void OnLiveViewClicked(object sender, EventArgs e)
        {
            image.IsVisible = false;
            feed.IsVisible = true;
        }

        private void OnDownloadClicked(object sender, EventArgs e)
        {
            DisplayAlert("Download", "Download button clicked", "OK");
        }

        private void OnDeleteClicked(object sender, EventArgs e)
        {
            DisplayAlert("Delete", "Delete button clicked", "OK");
        }

        // Solenoid control - toggled by switch
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

                // Update this with your actual Flask server address
                var response = await httpClient.PostAsync("http://192.168.176.237:5000/control_solenoid", content);
                var responseString = await response.Content.ReadAsStringAsync();

            }
            catch (Exception ex)
            {
                await DisplayAlert("Exception", ex.Message, "OK");
            }
        }

        private void OnCloseVideoClicked(object sender, EventArgs e)
        {
            image.IsVisible = true;
            feed.IsVisible = false;
        }

        private async Task LoadCapturedImages()
        {
            // Prevent concurrent loading
            lock (_loadingLock)
            {
                if (_isLoadingImages) return;
                _isLoadingImages = true;
            }

            try
            {
                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                string query = "SELECT id, image, timestamp FROM images ORDER BY timestamp DESC";
                using var cmd = new MySqlCommand(query, conn);
                using var reader = await cmd.ExecuteReaderAsync();

                // Clear existing items and tracking for initial load
                CapturedImagesContainer.Children.Clear();
                _loadedImageIds.Clear();

                int count = 0;
                while (await reader.ReadAsync())
                {
                    int id = reader.GetInt32("id");

                    // Skip if somehow already loaded (shouldn't happen in initial load, but safety check)
                    if (_loadedImageIds.Contains(id)) continue;

                    byte[] imageBytes = (byte[])reader["image"];
                    string timestamp = reader.GetDateTime("timestamp").ToString("yyyy-MM-dd HH:mm:ss");

                    // Update tracking variables
                    count++;
                    DateTime imageTimestamp = reader.GetDateTime("timestamp");
                    if (imageTimestamp > _lastImageTimestamp)
                    {
                        _lastImageTimestamp = imageTimestamp;
                    }

                    // Track this image as loaded
                    _loadedImageIds.Add(id);

                    // Create and add the image UI
                    var frame = CreateImageFrame(id, imageBytes, timestamp);
                    CapturedImagesContainer.Children.Add(frame);
                }

                // Update the last count
                _lastImageCount = count;
                System.Diagnostics.Debug.WriteLine($"Initial load completed: {count} images loaded");
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", ex.Message, "OK");
            }
            finally
            {
                lock (_loadingLock)
                {
                    _isLoadingImages = false;
                }
            }
        }

        // New method to load only new images
        private async Task LoadNewImagesOnly()
        {
            // Prevent concurrent loading
            lock (_loadingLock)
            {
                if (_isLoadingImages) return;
                _isLoadingImages = true;
            }

            try
            {
                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                // Get all images that we haven't loaded yet (by ID)
                string query;
                MySqlCommand cmd;

                if (_loadedImageIds.Count > 0)
                {
                    // Create a parameterized query to avoid SQL injection
                    var idParams = string.Join(",", _loadedImageIds.Select((id, index) => $"@id{index}"));
                    query = $@"SELECT id, image, timestamp FROM images 
                              WHERE id NOT IN ({idParams})
                              ORDER BY timestamp DESC";

                    cmd = new MySqlCommand(query, conn);
                    int paramIndex = 0;
                    foreach (var id in _loadedImageIds)
                    {
                        cmd.Parameters.AddWithValue($"@id{paramIndex}", id);
                        paramIndex++;
                    }
                }
                else
                {
                    // If no images loaded yet, get all images
                    query = "SELECT id, image, timestamp FROM images ORDER BY timestamp DESC";
                    cmd = new MySqlCommand(query, conn);
                }

                using var reader = await cmd.ExecuteReaderAsync();
                var newImages = new List<(int id, byte[] imageBytes, string timestamp, DateTime dateTime)>();

                while (await reader.ReadAsync())
                {
                    int id = reader.GetInt32("id");

                    // Double-check to prevent duplicates (race condition safety)
                    if (_loadedImageIds.Contains(id)) continue;

                    byte[] imageBytes = (byte[])reader["image"];
                    DateTime imageTimestamp = reader.GetDateTime("timestamp");
                    string timestamp = imageTimestamp.ToString("yyyy-MM-dd HH:mm:ss");

                    newImages.Add((id, imageBytes, timestamp, imageTimestamp));

                    // Update tracking variables
                    if (imageTimestamp > _lastImageTimestamp)
                    {
                        _lastImageTimestamp = imageTimestamp;
                    }

                    // Mark as loaded immediately to prevent duplicates
                    _loadedImageIds.Add(id);
                }

                // Add new images to the UI (newest first)
                foreach (var (id, imageBytes, timestamp, _) in newImages)
                {
                    // Final safety check before adding to UI
                    bool alreadyExists = CapturedImagesContainer.Children
                        .OfType<Frame>()
                        .Any(f => GetImageIdFromFrame(f) == id);

                    if (alreadyExists) continue;

                    var frame = CreateImageFrame(id, imageBytes, timestamp);

                    // Insert at the beginning to maintain newest-first order
                    CapturedImagesContainer.Children.Insert(0, frame);

                    // Add a subtle animation to highlight the new image
                    frame.Opacity = 0;
                    await frame.FadeTo(1, 500);

                    // Add a small delay between animations if multiple images
                    if (newImages.Count > 1)
                    {
                        await Task.Delay(100);
                    }
                }

                // Update count
                _lastImageCount = _loadedImageIds.Count;

                // Debug output
                if (newImages.Count > 0)
                {
                    System.Diagnostics.Debug.WriteLine($"Added {newImages.Count} new images");
                }
            }
            catch (Exception ex)
            {
                // Silent error handling for background refresh
                System.Diagnostics.Debug.WriteLine($"Load new images error: {ex.Message}");
            }
            finally
            {
                lock (_loadingLock)
                {
                    _isLoadingImages = false;
                }
            }
        }

        // Helper method to create image frame UI
        private Frame CreateImageFrame(int id, byte[] imageBytes, string timestamp)
        {
            // Convert image byte array to ImageSource
            ImageSource imageSource = ImageSource.FromStream(() => new MemoryStream(imageBytes));

            // Create UI dynamically
            var imageView = new Image
            {
                Source = imageSource,
                WidthRequest = 80,
                HeightRequest = 60,
                Aspect = Aspect.AspectFill
            };

            // Add tap gesture for full-screen view
            var tapGesture = new TapGestureRecognizer();
            tapGesture.Tapped += (s, e) => ShowFullScreenImage(imageSource);
            imageView.GestureRecognizers.Add(tapGesture);

            Grid.SetColumn(imageView, 0);

            var timestampLabel = new Label
            {
                Text = timestamp,
                FontSize = 14,
                TextColor = Colors.Red,
                VerticalOptions = LayoutOptions.Center,
                Margin = new Thickness(10, 0)
            };
            Grid.SetColumn(timestampLabel, 1);

            var downloadButton = new ImageButton
            {
                Source = "download.png",
                BackgroundColor = Colors.Transparent,
                WidthRequest = 30,
                HeightRequest = 30,
                Command = new Command(() => DownloadImage(imageBytes))
            };
            Grid.SetColumn(downloadButton, 2);

            var deleteButton = new ImageButton
            {
                Source = "delete.png",
                BackgroundColor = Colors.Transparent,
                WidthRequest = 30,
                HeightRequest = 30,
                Command = new Command(() => DeleteImage(id))
            };
            Grid.SetColumn(deleteButton, 3);

            var grid = new Grid
            {
                ColumnDefinitions =
                {
                    new ColumnDefinition { Width = GridLength.Auto },
                    new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) },
                    new ColumnDefinition { Width = GridLength.Auto },
                    new ColumnDefinition { Width = GridLength.Auto }
                },
                Children = { imageView, timestampLabel, downloadButton, deleteButton }
            };

            var frame = new Frame
            {
                BackgroundColor = Color.FromArgb("#F4F4F4"),
                CornerRadius = 15,
                Padding = 10,
                HasShadow = false,
                Content = grid
            };

            // Store the image ID in the frame's StyleId for easy retrieval
            frame.StyleId = id.ToString();

            return frame;
        }

        private async void DownloadImage(byte[] imageBytes)
        {
            try
            {
                // Create a unique filename with timestamp
                string fileName = $"captured_image_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";

                // Get the downloads directory path
                string downloadsPath = Path.Combine(FileSystem.Current.CacheDirectory, "Downloads");

                // Create downloads directory if it doesn't exist
                if (!Directory.Exists(downloadsPath))
                {
                    Directory.CreateDirectory(downloadsPath);
                }

                string filePath = Path.Combine(downloadsPath, fileName);

                // Write the image bytes to file
                await File.WriteAllBytesAsync(filePath, imageBytes);

#if ANDROID
                // For Android, copy to Pictures directory and add to media store
                await SaveToAndroidGallery(imageBytes, fileName);
#elif IOS
                // For iOS, save to photo library
                await SaveToIOSPhotoLibrary(imageBytes);
#else
                // For other platforms, show file location
                await DisplayAlert("Download Complete",
                    $"Image saved to: {filePath}", "OK");
#endif

            }
            catch (Exception ex)
            {
                await DisplayAlert("Download Error",
                    $"Failed to download image: {ex.Message}", "OK");
            }
        }

#if ANDROID
        private async Task SaveToAndroidGallery(byte[] imageBytes, string fileName)
        {
            try
            {
                var context = Platform.CurrentActivity ?? Android.App.Application.Context;

                // Use MediaStore for Android 10+ (API 29+)
                if (Android.OS.Build.VERSION.SdkInt >= Android.OS.BuildVersionCodes.Q)
                {
                    var contentValues = new Android.Content.ContentValues();
                    contentValues.Put(Android.Provider.MediaStore.Images.Media.InterfaceConsts.DisplayName, fileName);
                    contentValues.Put(Android.Provider.MediaStore.Images.Media.InterfaceConsts.MimeType, "image/jpeg");
                    contentValues.Put(Android.Provider.MediaStore.Images.Media.InterfaceConsts.RelativePath,
                        Android.OS.Environment.DirectoryPictures + "/CapturedImages");

                    var uri = context.ContentResolver.Insert(
                        Android.Provider.MediaStore.Images.Media.ExternalContentUri, contentValues);

                    if (uri != null)
                    {
                        using var outputStream = context.ContentResolver.OpenOutputStream(uri);
                        if (outputStream != null)
                        {
                            await outputStream.WriteAsync(imageBytes, 0, imageBytes.Length);
                            await outputStream.FlushAsync();
                        }

                        await DisplayAlert("Download Complete",
                            "Image saved to Pictures/CapturedImages folder", "OK");
                    }
                    else
                    {
                        throw new Exception("Failed to create media store entry");
                    }
                }
                else
                {
                    // Legacy approach for older Android versions
                    var picturesPath = Android.OS.Environment.GetExternalStoragePublicDirectory(
                        Android.OS.Environment.DirectoryPictures);

                    if (picturesPath != null)
                    {
                        var appFolder = new Java.IO.File(picturesPath, "CapturedImages");
                        if (!appFolder.Exists())
                        {
                            appFolder.Mkdirs();
                        }

                        var imageFile = new Java.IO.File(appFolder, fileName);
                        await File.WriteAllBytesAsync(imageFile.AbsolutePath, imageBytes);

                        // Notify media scanner
                        var intent = new Android.Content.Intent(Android.Content.Intent.ActionMediaScannerScanFile);
                        intent.SetData(Android.Net.Uri.FromFile(imageFile));
                        context.SendBroadcast(intent);

                        await DisplayAlert("Download Complete",
                            "Image saved to Pictures/CapturedImages folder", "OK");
                    }
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Download Error",
                    $"Failed to save to gallery: {ex.Message}", "OK");
            }
        }
#endif

#if IOS
        private async Task SaveToIOSPhotoLibrary(byte[] imageBytes)
        {
            try
            {
                var image = UIKit.UIImage.LoadFromData(Foundation.NSData.FromArray(imageBytes));
                if (image != null)
                {
                    Photos.PHPhotoLibrary.SharedPhotoLibrary.PerformChanges(() =>
                    {
                        Photos.PHAssetChangeRequest.FromImage(image);
                    }, (success, error) =>
                    {
                        MainThread.BeginInvokeOnMainThread(async () =>
                        {
                            if (success)
                            {
                                await DisplayAlert("Download Complete", 
                                    "Image saved to Photo Library", "OK");
                            }
                            else
                            {
                                await DisplayAlert("Download Error", 
                                    $"Failed to save to Photo Library: {error?.LocalizedDescription}", "OK");
                            }
                        });
                    });
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Download Error", 
                    $"Failed to save to Photo Library: {ex.Message}", "OK");
            }
        }
#endif

        private async void DeleteImage(int id)
        {
            bool confirm = await DisplayAlert("Delete", "Are you sure?", "Yes", "No");
            if (!confirm) return;

            try
            {
                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                var cmd = new MySqlCommand("DELETE FROM images WHERE id = @id", conn);
                cmd.Parameters.AddWithValue("@id", id);
                await cmd.ExecuteNonQueryAsync();

                // Remove from tracking
                _loadedImageIds.Remove(id);

                // Find and remove the corresponding UI element
                var frameToRemove = CapturedImagesContainer.Children
                    .OfType<Frame>()
                    .FirstOrDefault(f => GetImageIdFromFrame(f) == id);

                if (frameToRemove != null)
                {
                    // Animate out before removing
                    await frameToRemove.FadeTo(0, 200);
                    CapturedImagesContainer.Children.Remove(frameToRemove);
                }

                _lastImageCount = _loadedImageIds.Count;
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", ex.Message, "OK");
            }
        }

        // Helper method to get image ID from frame
        private int GetImageIdFromFrame(Frame frame)
        {
            if (frame?.StyleId != null && int.TryParse(frame.StyleId, out int id))
            {
                return id;
            }
            return -1; // Return -1 if ID cannot be determined
        }

        // Full-screen image viewing methods
        private void ShowFullScreenImage(ImageSource imageSource)
        {
            FullScreenImage.Source = imageSource;
            FullScreenOverlay.IsVisible = true;
        }

        private void OnFullScreenImageTapped(object sender, EventArgs e)
        {
            FullScreenOverlay.IsVisible = false;
        }

        private void OnCloseFullScreenClicked(object sender, EventArgs e)
        {
            FullScreenOverlay.IsVisible = false;
        }

        // Auto-refresh functionality
        private void StartAutoRefresh()
        {
            _refreshTimer = new System.Timers.Timer(5000); // Check every 5 seconds (increased from 3)
            _refreshTimer.Elapsed += async (sender, e) => await CheckForNewImages();
            _refreshTimer.AutoReset = true;
            _refreshTimer.Enabled = true;
            System.Diagnostics.Debug.WriteLine("Auto-refresh timer started");
        }

        private void StopAutoRefresh()
        {
            _refreshTimer?.Stop();
            _refreshTimer?.Dispose();
        }

        private async Task CheckForNewImages()
        {
            if (!_isPageActive) return;

            // Skip if already loading to prevent concurrent operations
            lock (_loadingLock)
            {
                if (_isLoadingImages) return;
            }

            try
            {
                using var conn = new MySqlConnection(_connStr);
                await conn.OpenAsync();

                // Get count of images not in our loaded set
                string countQuery;
                MySqlCommand countCmd;

                if (_loadedImageIds.Count > 0)
                {
                    var idParams = string.Join(",", _loadedImageIds.Select((id, index) => $"@id{index}"));
                    countQuery = $"SELECT COUNT(*) as new_count FROM images WHERE id NOT IN ({idParams})";
                    countCmd = new MySqlCommand(countQuery, conn);

                    int paramIndex = 0;
                    foreach (var id in _loadedImageIds)
                    {
                        countCmd.Parameters.AddWithValue($"@id{paramIndex}", id);
                        paramIndex++;
                    }
                }
                else
                {
                    countQuery = "SELECT COUNT(*) as new_count FROM images";
                    countCmd = new MySqlCommand(countQuery, conn);
                }

                using var reader = await countCmd.ExecuteReaderAsync();

                if (await reader.ReadAsync())
                {
                    int newImageCount = reader.GetInt32("new_count");

                    // Debug output
                    System.Diagnostics.Debug.WriteLine($"Checking for new images: {newImageCount} new images found");

                    if (newImageCount > 0)
                    {
                        // Update UI on main thread - load only new images
                        MainThread.BeginInvokeOnMainThread(async () =>
                        {
                            await LoadNewImagesOnly();
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                // Silent error handling for background refresh
                System.Diagnostics.Debug.WriteLine($"Auto-refresh error: {ex.Message}");
            }
        }



        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            _isPageActive = false;
            StopAutoRefresh();
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