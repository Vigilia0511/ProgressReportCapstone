public class User
{
    public string Username { get; set; }

    public string Status { get; set; } // "Approved" or "Not Approved"

    public string StatusIcon => Status == "Approved" ? "check.png" : "cross.png";

    public string ActionText => Status == "Approved" ? "Disallow" : "Approve";

    public Color ActionColor => Status == "Approved" ? Colors.Blue : Colors.Green;
}
