<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

include 'dbconnect.php';

// Latest 10 notifications
$queryLatest = "SELECT notify AS message, timestamp FROM logs ORDER BY timestamp DESC LIMIT 10";
$resultLatest = mysqli_query($conn, $queryLatest);
$latestNotifications = [];
if ($resultLatest) {
    while ($row = mysqli_fetch_assoc($resultLatest)) {
        $latestNotifications[] = $row;
    }
}

// All notifications
$queryAll = "SELECT notify AS message, timestamp FROM logs ORDER BY timestamp DESC";
$resultAll = mysqli_query($conn, $queryAll);
$allNotifications = [];
if ($resultAll) {
    while ($row = mysqli_fetch_assoc($resultAll)) {
        $allNotifications[] = $row;
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Captured Images</title>
  <link rel="stylesheet" href="css/styles.css">
  <style>
  #liveViewContainer {
      display: none;
      position: fixed;
      top: 0; left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(255, 255, 255, 0.85);
      z-index: 10000;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      gap: 15px;
      padding: 20px;
      box-sizing: border-box;
    }

    #liveViewContainer h2 {
      color: #333;
      margin: 0;
      font-size: 1.8rem;
    }

    #liveViewContainer img {
      max-width: 95vw;
      max-height: 80vh;
      border-radius: 8px;
      box-shadow: 0 0 20px rgba(0,0,0,0.3);
    }

    #closeLiveView {
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ff4444;
      color: white;
      border: none;
      font-size: 2rem;
      width: 44px;
      height: 44px;
      border-radius: 50%;
      cursor: pointer;
      z-index: 11000;
      line-height: 40px;
      text-align: center;
      padding: 0;
      transition: background 0.3s ease;
    }

    #closeLiveView:hover {
      background: #ff0000;
    }

    .controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px;
    }

    .toggle-label {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .live-view-btn {
      padding: 8px 14px;
      background-color:rgb(255, 0, 34);
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
  </style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="logo"><img src="image/logo.png" alt="Logo" /></div>
  <img src="image/menu.png" class="menu-icon" alt="Menu" id="menuIcon" />
</div>

<!-- CONTROLS -->
<div class="controls">
  <div class="toggle-label">
    <label>OPEN DOOR LOCK</label>
    <label class="toggle-switch">
      <input type="checkbox" />
      <span class="slider"></span>
    </label>
  </div>
  <button class="live-view-btn">LIVE VIEW</button>
</div>


<!-- LIVE VIEW -->
<div id="liveViewContainer">
    <button id="closeLiveView" aria-label="Close Live View">×</button>
    <img src="http://192.168.176.237:5000/video_feed" alt="Live Video Feed" title="Live Video Feed" />
  </div>

<!-- NOTIFICATION AREA -->
<div class="notification-container" id="notificationContainer">
  <h2>NOTIFICATION AREA</h2>
  <div class="notification-list">
    <?php foreach ($latestNotifications as $notif):
      $msg = strtolower($notif['message']);
      $statusClass = (strpos($msg, 'granted') !== false || strpos($msg, 'registered') !== false) ? 'success' :
                     ((strpos($msg, 'denied') !== false || strpos($msg, 'failed') !== false)  ? 'error' : '');
    ?>
      <div class="notification-item <?= $statusClass ?>">
        <?= htmlspecialchars($notif['message']) ?> at <?= htmlspecialchars($notif['timestamp']) ?>
      </div>
    <?php endforeach; ?>
  </div>
</div>

<!-- VIEW ALL BUTTON -->
<div style="text-align: center; margin: 20px 0;">
  <button id="viewAllBtn" class="view-all-btn">View All Notifications</button>
</div>

<!-- VIEW ALL NOTIFICATIONS -->
<div id="allNotifications" style="display: none; padding: 20px;">
  <button id="closeAllNotifications">x</button>
  <div class="notification-grid">
    <div class="notification-category" id="fingerprintNotifications"><h3>Fingerprint</h3></div>
    <div class="notification-category" id="voiceNotifications"><h3>Voice</h3></div>
    <div class="notification-category" id="pinNotifications"><h3>PIN</h3></div>
    <div class="notification-category" id="faceNotifications"><h3>Face</h3></div>
  </div>
</div>

<!-- SIDEBAR -->
<div class="overlay" id="overlay"></div>
<nav class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <img src="image/profile.png" alt="User Icon" class="user-icon" />
    <p>Welcome, <strong>ADMIN</strong></p>
  </div>
  <a href="allnotif.php" class="btn-link"><img src="image/home.png" class="sidebar-icon" /> Home</a>
  <a href="approval.php" class="btn-link"><img src="image/approval.png" class="sidebar-icon" /> Approval</a>
  <a href="images.php" class="btn-link"><img src="image/picture.png" class="sidebar-icon" /> Image</a>
  <button id="settingsToggle" aria-expanded="false" aria-controls="settingsDropdown">
    <img src="image/settings.png" class="sidebar-icon" /> Settings
    <svg class="dropdown-arrow" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
      <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/>
    </svg>
  </button>
  <div class="settings-dropdown" id="settingsDropdown">
    <div class="toggle-label">
      <label for="darkModeToggle">Dark Mode</label>
      <label class="toggle-switch">
        <input type="checkbox" id="darkModeToggle" />
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-label">
      <label for="biometricToggle">Enable Biometric</label>
      <label class="toggle-switch">
        <input type="checkbox" id="biometricToggle" checked />
        <span class="slider"></span>
      </label>
    </div>
  </div>
  <button id="logoutBtn"><img src="image/logout.png" class="sidebar-icon" /> Logout</button>
</nav>

<!-- JS SCRIPTS -->
<script>
  const notifications = <?= json_encode($allNotifications) ?>;

  function getCategory(message) {
    message = message.toLowerCase();
    if (message.includes('fingerprint')) return 'fingerprint';
    if (message.includes('voice')) return 'voice';
    if (message.includes('pin')) return 'pin';
    if (message.includes('face')) return 'face';
    return 'other';
  }

  function getStatusClass(message) {
    message = message.toLowerCase();
    if (message.includes('granted') || message.includes('registered') || message.includes('successfully')) return 'success';
    if (message.includes('denied') || message.includes('failed')) return 'error';
    return '';
  }

  function populateNotifications() {
    const categories = {
      fingerprint: document.getElementById('fingerprintNotifications'),
      voice: document.getElementById('voiceNotifications'),
      pin: document.getElementById('pinNotifications'),
      face: document.getElementById('faceNotifications')
    };

    for (const key in categories) {
      const container = categories[key];
      while (container.children.length > 1) {
        container.removeChild(container.lastChild);
      }
    }

    notifications.forEach(n => {
      const category = getCategory(n.message);
      if (!categories[category]) return;
      const div = document.createElement('div');
      div.className = 'notification-item ' + getStatusClass(n.message);
      div.textContent = `${n.message} at ${n.timestamp}`;
      categories[category].appendChild(div);
    });
  }

  document.getElementById('viewAllBtn').addEventListener('click', () => {
    populateNotifications();
    document.getElementById('notificationContainer').style.display = 'none';
    document.getElementById('allNotifications').style.display = 'block';
        document.getElementById('viewAllBtn').style.display = 'none';
  });

document.getElementById('closeAllNotifications').addEventListener('click', () => {
    document.getElementById('allNotifications').style.display = 'none';
    document.getElementById('notificationContainer').style.display = 'block';
    document.getElementById('viewAllBtn').style.display = 'inline-block';
});


  const liveViewBtn = document.querySelector('.live-view-btn');
    const liveViewContainer = document.getElementById('liveViewContainer');
    const closeLiveView = document.getElementById('closeLiveView');

    liveViewBtn.addEventListener('click', () => {
      liveViewContainer.style.display = 'flex';
    });

    closeLiveView.addEventListener('click', () => {
      liveViewContainer.style.display = 'none';
      document.getElementById('notificationContainer').style.display = 'block';
    });

        
        

  document.getElementById('menuIcon').addEventListener('click', () => {
    document.getElementById('sidebar').classList.add('visible');
    document.getElementById('overlay').style.display = 'block';
  });

  document.getElementById('overlay').addEventListener('click', () => {
    document.getElementById('sidebar').classList.remove('visible');
    document.getElementById('overlay').style.display = 'none';
  });

  document.getElementById('darkModeToggle').addEventListener('change', (e) => {
    document.body.classList.toggle('dark', e.target.checked);
  });

  const settingsToggle = document.getElementById('settingsToggle');
  const settingsDropdown = document.getElementById('settingsDropdown');
  const dropdownArrow = settingsToggle.querySelector('.dropdown-arrow');

  settingsToggle.addEventListener('click', () => {
    const isOpen = settingsDropdown.classList.toggle('open');
    dropdownArrow.classList.toggle('open', isOpen);
    settingsToggle.setAttribute('aria-expanded', isOpen);
  });

  document.getElementById('logoutBtn').addEventListener('click', () => {
    if (confirm("Are you sure you want to log out?")) {
      window.location.href = "logout.php";
    }
  });
</script>
</body>
</html>
