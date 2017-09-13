var updateInterval = 1000;

var updateStatus = function(data) {
    document.getElementById('status_alive').innerHTML = data.alive;
    document.getElementById('status_state').innerHTML = data.state;
    document.getElementById('status_speed').innerHTML = data.speed;    
    document.getElementById('status_posx').innerHTML = data.posx;
    document.getElementById('status_posy').innerHTML = data.posy;
    document.getElementById('status_posz').innerHTML = data.posz;
    document.getElementById('status_pose').innerHTML = data.pose;
    document.getElementById('status_vacuum').innerHTML = data.vacuum;
    document.getElementById('status_light').innerHTML = data.light;
    buttonsEnabled(true);
}

var fetchStatus = function() {
    fetch('/status.json').then(function(resp) {
       return resp.json();
    }).then(function(data) {
       updateStatus(data);
    });
    setTimeout(fetchStatus, updateInterval);
}

var buttonAction = function(method) {
    buttonsEnabled(false);
    fetch('/' + method, { method: 'POST' });
}

var buttonsEnabled = function(state) {
    var e = document.getElementsByTagName('button');
    for (i = 0; i < e.length; i++) {
        e[i].disabled = state ? false : true;
    }
}

fetchStatus();
