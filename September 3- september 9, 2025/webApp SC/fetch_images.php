<?php
header('Content-Type: application/json');

$host = "localhost";
$username = "root";
$password = "";
$database = "smartdb";

$conn = new mysqli($host, $username, $password, $database);
if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "DB connection failed: " . $conn->connect_error]);
    exit;
}

// If 'delete' param is set, process deletion
if (isset($_GET['delete'])) {
    $id = intval($_GET['delete']);
    $stmt = $conn->prepare("DELETE FROM images WHERE id = ?");
    if (!$stmt) {
        http_response_code(500);
        echo json_encode(["success" => false, "message" => "Prepare failed: " . $conn->error]);
        exit;
    }
    $stmt->bind_param("i", $id);
    if ($stmt->execute()) {
        if ($stmt->affected_rows > 0) {
            echo json_encode(["success" => true]);
        } else {
            echo json_encode(["success" => false, "message" => "Image not found or already deleted"]);
        }
    } else {
        http_response_code(500);
        echo json_encode(["success" => false, "message" => "Execute failed: " . $stmt->error]);
    }
    $stmt->close();
    $conn->close();
    exit;
}

// Otherwise, fetch and return images
$sql = "SELECT id, image, timestamp FROM images ORDER BY timestamp DESC";
$result = $conn->query($sql);

$images = [];

if ($result) {
    while ($row = $result->fetch_assoc()) {
        $base64Image = 'data:image/jpeg;base64,' . base64_encode($row['image']);
        $images[] = [
            "id" => $row['id'],
            "image" => $base64Image,
            "timestamp" => $row["timestamp"]
        ];
    }
}

echo json_encode($images);

$conn->close();
?>
