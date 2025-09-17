using MySql.Data.MySqlClient;
using System.Collections.ObjectModel;

namespace testdatabase;

public partial class UserListPage : ContentPage
{
    public ObservableCollection<UserData> Users { get; set; } = new();

    public UserListPage()
    {
        InitializeComponent();
        BindingContext = this;
        LoadUsers();
    }

    private async void LoadUsers()
    {
        string connStr = "server=192.168.1.149;user=root;password=oneinamillion;database=smartdb;";
        using MySqlConnection conn = new(connStr);

        try
        {
            await conn.OpenAsync();

            string query = "SELECT username, timestamp FROM users ORDER BY timestamp DESC";
            using MySqlCommand cmd = new(query, conn);
            using MySqlDataReader reader = (MySqlDataReader)await cmd.ExecuteReaderAsync();

            Users.Clear(); // Clear old list if reloading
            while (await reader.ReadAsync())
            {
                Users.Add(new UserData
                {
                    Username = reader.GetString("username"),
                    CreatedAt = reader.GetDateTime("timestamp").ToString("yyyy-MM-dd HH:mm:ss")
                });
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("Database Error", ex.Message, "OK");
        }
    }
}

public class UserData
{
    public string Username { get; set; }
    public string CreatedAt { get; set; }
}
