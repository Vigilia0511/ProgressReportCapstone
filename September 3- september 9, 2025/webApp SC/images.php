<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Captured Images</title>
  <link rel="stylesheet" href="css/styles1.css" />

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
  <div class="header">
    <div class="logo">
      <img src="image/logo.png" alt="Logo" />
    </div>
    <img src="image/menu.png" class="menu-icon" alt="Menu" id="menuIcon" />
  </div>

  <div class="controls">
    <div class="toggle-label">
      <label>OPEN DOOR LOCK</label>
      <label class="toggle-switch">
        <input type="checkbox" id="doorLockToggle" />
        <span class="slider"></span>
      </label>
    </div>
    <button class="live-view-btn">LIVE VIEW</button>
  </div>

  <div id="liveViewContainer">
    <button id="closeLiveView" aria-label="Close Live View">×</button>
    <img src="http://192.168.176.237:5000/video_feed" alt="Live Video Feed" title="Live Video Feed" />
  </div>

  <div class="section-title">CAPTURED IMAGES</div>
  <div class="image-list" id="imageList"></div>

  <div id="fullscreenOverlay" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.8); justify-content:center; align-items:center;">
    <img id="fullscreenImage" src="" style="max-width:90vw; max-height:90vh;" />
  </div>

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

  <script>
    document.getElementById('logoutBtn').addEventListener('click', () => {
      if (confirm("Are you sure you want to log out?")) {
        window.location.href = "logout.php";
      }
    });

    const menuIcon = document.getElementById('menuIcon');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('overlay');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsDropdown = document.getElementById('settingsDropdown');
    const dropdownArrow = settingsToggle.querySelector('.dropdown-arrow');

    menuIcon.addEventListener('click', () => {
      sidebar.classList.add('visible');
      overlay.style.display = 'block';
    });

    overlay.addEventListener('click', () => {
      sidebar.classList.remove('visible');
      overlay.style.display = 'none';
    });

    darkModeToggle.addEventListener('change', (e) => {
      document.body.classList.toggle('dark', e.target.checked);
    });

    settingsToggle.addEventListener('click', () => {
      const isOpen = settingsDropdown.classList.toggle('open');
      dropdownArrow.classList.toggle('open', isOpen);
      settingsToggle.setAttribute('aria-expanded', isOpen);
    });

    const doorLockToggle = document.getElementById('doorLockToggle');
    doorLockToggle.addEventListener('change', async () => {
      const state = doorLockToggle.checked ? 'on' : 'off';
      try {
        const formData = new URLSearchParams();
        formData.append('switch', state);

        const response = await fetch('http://192.168.176.237:5000/control_solenoid', {
          method: 'POST',
          mode: 'cors',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData
        });

        if (!response.ok) throw new Error(`Server responded with status ${response.status}`);

        const jsonResponse = await response.json();
        console.log('Solenoid control response:', jsonResponse);

        if (jsonResponse.status !== 'success') {
          throw new Error(jsonResponse.message || 'Unknown error from server');
        }

      } catch (error) {
        alert(`Error toggling door lock:\n${error.message}`);
        console.error(error);
        doorLockToggle.checked = !doorLockToggle.checked;
      }
    });

    const liveViewBtn = document.querySelector('.live-view-btn');
    const liveViewContainer = document.getElementById('liveViewContainer');
    const closeLiveView = document.getElementById('closeLiveView');

    liveViewBtn.addEventListener('click', () => {
      liveViewContainer.style.display = 'flex';
    });

    closeLiveView.addEventListener('click', () => {
      liveViewContainer.style.display = 'none';
    });

    fetch('fetch_images.php')
      .then(response => response.json())
      .then(data => {
        const imageList = document.getElementById('imageList');
        imageList.innerHTML = '';
        data.forEach(item => {
          const card = document.createElement('div');
          card.className = 'image-card';
          card.innerHTML = `
            <img src="${item.image}" alt="Captured Image" />
            <div class="timestamp">${item.timestamp}</div>
            <button class="icon-button download-btn" title="Download"><img src="image/download.png" alt="Download" /></button>
            <button class="icon-button delete-btn" title="Delete"><img src="image/delete.png" alt="Delete" /></button>
          `;

          card.querySelector('img').addEventListener('click', () => {
            const fullscreenOverlay = document.getElementById('fullscreenOverlay');
            const fullscreenImage = document.getElementById('fullscreenImage');
            fullscreenImage.src = item.image;
            fullscreenOverlay.style.display = 'flex';
          });

          card.querySelector('.download-btn').addEventListener('click', () => {
            const a = document.createElement('a');
            a.href = item.image;
            a.download = `captured_${item.id}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
          });

          card.querySelector('.delete-btn').addEventListener('click', () => {
            if (confirm('Are you sure you want to delete this image?')) {
              fetch(`fetch_images.php?delete=${encodeURIComponent(item.id)}`, { method: 'GET' })
              .then(response => response.json())
              .then(result => {
                if (result.success) {
                  card.remove();
                  alert('Image deleted successfully.');
                } else {
                  alert('Failed to delete image: ' + (result.message || 'Unknown error'));
                }
              })
              .catch(err => {
                alert('Error deleting image: ' + err.message);
              });
            }
          });

          imageList.appendChild(card);
        });
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        document.getElementById('imageList').innerHTML = '<p>Error loading images.</p>';
      });

    document.getElementById('fullscreenOverlay').addEventListener('click', (e) => {
      if (e.target.id === 'fullscreenOverlay') {
        e.target.style.display = 'none';
        document.getElementById('fullscreenImage').src = '';
      }
    });
  </script>
</body>
</html>
