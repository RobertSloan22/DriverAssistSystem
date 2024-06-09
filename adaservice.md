[Unit]
Description=Driver Assist Program
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /jetson-inference/build/aarch64/bin/DriverAssistSystem/ADAS.py
Restart=on-failure
User=nvidia
WorkingDirectory=/jetson-inference/build/aarch64/bin/DriverAssistSystem

[Install]
WantedBy=multi-user.target
