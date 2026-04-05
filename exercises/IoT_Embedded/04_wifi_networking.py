"""
Exercises for Lesson 04: WiFi Networking
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Demonstrates TCP/UDP socket programming, HTTP communication, and
network scanning -- all runnable without real IoT hardware.
"""

import socket
import threading
import time
import json
import csv
import os
import random
import re
from datetime import datetime
from collections import deque


# === Exercise 1: WiFi Signal Monitor ===
# Problem: Continuously monitor WiFi signal strength and connection quality.
# Parse iwconfig output, log to CSV, warn when signal drops, track disconnections.

def exercise_1():
    """Solution: WiFi signal monitor with CSV logging and disconnection tracking.

    On a real Raspberry Pi, iwconfig wlan0 returns lines like:
        wlan0     IEEE 802.11  ESSID:"MyNetwork"
                  Mode:Managed  Frequency:5.18 GHz  Access Point: AA:BB:CC:DD:EE:FF
                  Link Quality=70/70  Signal level=-40 dBm
    We simulate this output for portability.
    """

    def simulate_iwconfig():
        """Simulate iwconfig wlan0 output.

        Real implementation:
            result = subprocess.check_output(['iwconfig', 'wlan0'],
                                            stderr=subprocess.DEVNULL,
                                            universal_newlines=True)
            return result
        """
        # Simulate occasional disconnection (no SSID)
        if random.random() < 0.15:
            return 'wlan0     IEEE 802.11  ESSID:off/any\n          No link'

        signal = random.randint(-80, -30)
        quality = max(0, min(70, 70 + signal + 40))
        return (
            f'wlan0     IEEE 802.11  ESSID:"HomeNetwork"\n'
            f'          Link Quality={quality}/70  Signal level={signal} dBm'
        )

    def parse_iwconfig(output):
        """Parse SSID, signal level, and link quality from iwconfig output."""
        ssid_match = re.search(r'ESSID:"(.+?)"', output)
        signal_match = re.search(r'Signal level=(-?\d+) dBm', output)
        quality_match = re.search(r'Link Quality=(\d+)/(\d+)', output)

        ssid = ssid_match.group(1) if ssid_match else None
        signal = int(signal_match.group(1)) if signal_match else None
        quality = f"{quality_match.group(1)}/{quality_match.group(2)}" if quality_match else None

        return ssid, signal, quality

    csv_file = "/tmp/wifi_signal_log.csv"
    disconnected_at = None
    total_downtime = 0.0

    # Initialize CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ssid", "signal_dbm", "quality"])

    print("  WiFi Signal Monitor (10 readings)\n")

    for i in range(10):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = simulate_iwconfig()
        ssid, signal, quality = parse_iwconfig(output)

        if ssid is None:
            # Disconnected
            if disconnected_at is None:
                disconnected_at = datetime.now()
                print(f"    [{ts}] ** DISCONNECTED ** (no SSID)")
            else:
                print(f"    [{ts}] Still disconnected...")
        else:
            if disconnected_at is not None:
                # Just reconnected
                outage = (datetime.now() - disconnected_at).total_seconds()
                total_downtime += outage
                print(f"    [{ts}] RECONNECTED after {outage:.1f}s downtime")
                disconnected_at = None

            # Log to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, ssid, signal, quality])

            # Warn if signal is weak
            warning = " ** WEAK SIGNAL **" if signal and signal < -70 else ""
            print(f"    [{ts}] SSID={ssid}, Signal={signal} dBm, "
                  f"Quality={quality}{warning}")

        time.sleep(0.5)

    print(f"\n  Total downtime: {total_downtime:.1f}s")
    print(f"  Log saved to: {csv_file}")

    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)


# === Exercise 2: Multi-Client TCP Sensor Server ===
# Problem: TCP server accepts concurrent sensor clients, maintains latest
# readings per sensor_id, prints summary every 30s. Client sends 10 readings.

def exercise_2():
    """Solution: Multi-client TCP sensor server and client.

    TCP provides reliable, ordered delivery -- ideal for sensor data that
    must not be lost. Each client runs in its own thread on the server side.
    """

    HOST = "127.0.0.1"
    PORT = 18888
    latest_readings = {}  # sensor_id -> latest reading
    readings_lock = threading.Lock()

    class SensorTCPServer:
        """TCP server that accepts JSON sensor data from multiple clients."""

        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.server_socket = None
            self.running = False

        def handle_client(self, client_socket, address):
            """Handle one sensor client connection in a dedicated thread."""
            try:
                while self.running:
                    data = client_socket.recv(4096)
                    if not data:
                        break

                    message = json.loads(data.decode("utf-8"))
                    sensor_id = message.get("sensor_id", "unknown")
                    value = message.get("value")
                    unit = message.get("unit", "")

                    # Store latest reading (thread-safe)
                    with readings_lock:
                        latest_readings[sensor_id] = {
                            "value": value,
                            "unit": unit,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "address": str(address),
                        }

                    # Send acknowledgment
                    ack = json.dumps({"status": "ok", "sensor_id": sensor_id})
                    client_socket.send(ack.encode("utf-8"))

            except (json.JSONDecodeError, ConnectionResetError) as e:
                pass
            finally:
                client_socket.close()

        def print_summary(self):
            """Print status summary of all sensors."""
            with readings_lock:
                if latest_readings:
                    print("\n    --- Sensor Status Summary ---")
                    for sid, info in sorted(latest_readings.items()):
                        print(f"      {sid}: {info['value']} {info['unit']} "
                              f"(at {info['timestamp']}, from {info['address']})")
                    print()

        def start(self, run_duration=5):
            """Start server for a limited duration (demo)."""
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)  # Allow periodic checks
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            print(f"    Server listening on {self.host}:{self.port}")

            end_time = time.time() + run_duration
            while self.running and time.time() < end_time:
                try:
                    client_socket, address = self.server_socket.accept()
                    t = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue

            self.running = False
            self.server_socket.close()
            self.print_summary()

    class SensorTCPClient:
        """TCP client that sends simulated sensor readings."""

        def __init__(self, host, port, sensor_id, unit="C"):
            self.host = host
            self.port = port
            self.sensor_id = sensor_id
            self.unit = unit

        def send_readings(self, count=10, interval=0.3):
            """Send count readings at the given interval."""
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))

                for i in range(count):
                    reading = {
                        "sensor_id": self.sensor_id,
                        "value": round(random.uniform(20.0, 30.0), 1),
                        "unit": self.unit,
                    }
                    sock.send(json.dumps(reading).encode("utf-8"))

                    # Wait for ack
                    ack = sock.recv(1024).decode("utf-8")

                    if i < count - 1:
                        time.sleep(interval)

                sock.close()
            except ConnectionRefusedError:
                pass

    print("  Multi-Client TCP Sensor Server\n")

    # Start server in background
    server = SensorTCPServer(HOST, PORT)
    server_thread = threading.Thread(target=server.start, args=(5,), daemon=True)
    server_thread.start()

    time.sleep(0.5)  # Let server start

    # Start two sensor clients
    client1 = SensorTCPClient(HOST, PORT, "temp_sensor_01", "C")
    client2 = SensorTCPClient(HOST, PORT, "humidity_sensor_01", "%")

    t1 = threading.Thread(target=client1.send_readings, args=(5, 0.3), daemon=True)
    t2 = threading.Thread(target=client2.send_readings, args=(5, 0.3), daemon=True)

    print("    Starting client 1: temp_sensor_01")
    t1.start()
    print("    Starting client 2: humidity_sensor_01")
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)
    server_thread.join(timeout=6)

    print("  Server and clients finished.")


# === Exercise 3: HTTP Sensor Reporter with Retry ===
# Problem: Production-grade sensor data sender with persistent queue,
# exponential backoff, batch flush, and rotating log file.

def exercise_3():
    """Solution: HTTP sensor reporter with queue, backoff, and batch flush.

    In production this sends to a real HTTP endpoint. Here we simulate
    the server responses to demonstrate the retry and batching logic.
    """

    import logging
    from logging.handlers import RotatingFileHandler

    log_file = "/tmp/sensor_reporter.log"

    # Set up rotating logger
    logger = logging.getLogger("sensor_reporter")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=10_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    class RobustSensorSender:
        """HTTP sensor reporter with persistent queue and exponential backoff.

        Key resilience features:
        1. Local deque buffer (max 500) prevents data loss during outages
        2. Exponential backoff (5s -> 10s -> 20s -> ... -> 300s) avoids
           hammering a struggling server
        3. Background batch flush thread drains the queue when connectivity returns
        """

        def __init__(self, api_url, device_id, max_queue=500):
            self.api_url = api_url
            self.device_id = device_id
            self.queue = deque(maxlen=max_queue)
            self.backoff = 5  # Initial backoff in seconds
            self.max_backoff = 300  # 5 minutes max
            self.consecutive_failures = 0

        def simulate_server_response(self):
            """Simulate HTTP POST outcome. 70% success, 30% failure."""
            return random.random() < 0.7

        def send_reading(self, data):
            """Attempt to send a single reading via HTTP POST.

            Real implementation:
                response = requests.post(self.api_url, json=data, timeout=5)
                return response.status_code == 200
            """
            success = self.simulate_server_response()
            if success:
                self.consecutive_failures = 0
                self.backoff = 5  # Reset backoff on success
                logger.info(f"Sent: {data}")
                return True
            else:
                self.consecutive_failures += 1
                # Exponential backoff: 5, 10, 20, 40, 80, 160, 300 (capped)
                self.backoff = min(self.backoff * 2, self.max_backoff)
                logger.warning(
                    f"Send failed (attempt {self.consecutive_failures}), "
                    f"next retry in {self.backoff}s"
                )
                self.queue.appendleft(data)  # Put back in queue
                return False

        def add_reading(self, data):
            """Add a reading to the queue."""
            self.queue.append(data)

        def flush_batch(self, batch_size=10):
            """Flush up to batch_size readings from the queue."""
            sent = 0
            for _ in range(min(batch_size, len(self.queue))):
                if not self.queue:
                    break
                data = self.queue.popleft()
                if self.send_reading(data):
                    sent += 1
                else:
                    break  # Stop flushing on first failure (backoff)

            if sent > 0:
                logger.info(f"Batch flush: {sent} readings sent, "
                           f"{len(self.queue)} remaining in queue")
            return sent

    print("  HTTP Sensor Reporter with Retry\n")

    sender = RobustSensorSender(
        api_url="https://api.example.com/sensors/data",
        device_id="RPi_001",
    )

    # Simulate collecting 15 readings
    print("    Collecting 15 sensor readings...")
    for i in range(15):
        reading = {
            "device_id": sender.device_id,
            "temperature": round(random.uniform(20, 30), 1),
            "humidity": round(random.uniform(40, 70), 1),
            "timestamp": datetime.now().isoformat(),
        }
        sender.add_reading(reading)

    print(f"    Queue size: {len(sender.queue)}\n")

    # Flush in batches
    print("    Flushing in batches of 5...")
    total_sent = 0
    for batch_num in range(1, 5):
        sent = sender.flush_batch(batch_size=5)
        total_sent += sent
        print(f"      Batch {batch_num}: sent {sent}, queue remaining: {len(sender.queue)}")
        if len(sender.queue) == 0:
            break
        time.sleep(0.3)

    print(f"\n    Total sent: {total_sent}, remaining in queue: {len(sender.queue)}")
    print(f"    Current backoff: {sender.backoff}s")

    # Show log contents
    logger.handlers[0].flush()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
        print(f"\n    --- Log file ({len(lines)} entries) ---")
        for line in lines[-5:]:  # Show last 5 entries
            print(f"      {line.strip()}")
        os.remove(log_file)

    # Clean up logger
    logger.removeHandler(handler)
    handler.close()


# === Exercise 4: Network Device Scanner with Service Detection ===
# Problem: Scan subnet, check IoT ports, reverse DNS, formatted table,
# periodic scan with new device alerts.

def exercise_4():
    """Solution: Network device scanner with service detection.

    Uses socket connections to probe common IoT ports. On a real network
    this discovers actual devices; here we simulate for demonstration.
    """

    class IoTDeviceScanner:
        """Scan a subnet for IoT devices and detect their services."""

        COMMON_PORTS = {
            22: "SSH",
            23: "Telnet",
            80: "HTTP",
            443: "HTTPS",
            1883: "MQTT",
            8883: "MQTT-TLS",
            8080: "HTTP-Alt",
        }

        def __init__(self, network="192.168.1.0/24"):
            self.network = network
            self.previous_devices = set()

        def simulate_scan(self):
            """Simulate network scan results.

            Real implementation would use:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((str(ip), port))
                is_open = (result == 0)
            """
            devices = {
                "192.168.1.1": {
                    "hostname": "router.local",
                    "services": {80: "HTTP", 443: "HTTPS", 22: "SSH"},
                },
                "192.168.1.10": {
                    "hostname": "raspberrypi.local",
                    "services": {22: "SSH", 1883: "MQTT", 8080: "HTTP-Alt"},
                },
                "192.168.1.20": {
                    "hostname": "esp32-sensor.local",
                    "services": {80: "HTTP", 1883: "MQTT"},
                },
                "192.168.1.30": {
                    "hostname": None,  # No reverse DNS
                    "services": {23: "Telnet", 80: "HTTP"},
                },
            }

            # Randomly add a "new" device 50% of the time
            if random.random() > 0.5:
                new_ip = f"192.168.1.{random.randint(40, 50)}"
                devices[new_ip] = {
                    "hostname": None,
                    "services": {80: "HTTP"},
                }

            return devices

        def resolve_hostname(self, ip):
            """Attempt reverse DNS lookup.

            Real implementation:
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                except socket.herror:
                    hostname = None
            """
            # Handled in simulate_scan for this demo
            pass

        def scan_and_report(self, scan_number=1):
            """Run one scan cycle, detect new/disappeared devices, print table."""
            devices = self.simulate_scan()
            current_ips = set(devices.keys())

            # Detect changes
            new_devices = current_ips - self.previous_devices
            disappeared = self.previous_devices - current_ips

            print(f"\n    --- Scan #{scan_number} ({self.network}) ---\n")

            # Print formatted table
            print(f"    {'IP Address':<18} {'Hostname':<25} {'Open Services'}")
            print(f"    {'-'*18} {'-'*25} {'-'*30}")

            for ip in sorted(devices.keys()):
                info = devices[ip]
                hostname = info["hostname"] or "(no DNS)"
                services = ", ".join(
                    f"{port}/{name}" for port, name in sorted(info["services"].items())
                )
                marker = " ** NEW **" if ip in new_devices else ""
                print(f"    {ip:<18} {hostname:<25} {services}{marker}")

            if disappeared:
                print(f"\n    Disappeared: {', '.join(sorted(disappeared))}")

            if new_devices:
                print(f"\n    *** ALERT: {len(new_devices)} new device(s) detected! ***")
                for ip in sorted(new_devices):
                    print(f"        -> {ip}")

            self.previous_devices = current_ips
            return devices

    print("  Network Device Scanner with Service Detection\n")

    scanner = IoTDeviceScanner("192.168.1.0/24")

    # Run 3 scan cycles (simulating 5-minute intervals)
    for scan_num in range(1, 4):
        scanner.scan_and_report(scan_number=scan_num)
        if scan_num < 3:
            time.sleep(0.5)


# === Exercise 5: TCP vs UDP Protocol Benchmark ===
# Problem: Compare TCP and UDP round-trip times (1000 messages of 100 bytes).

def exercise_5():
    """Solution: TCP vs UDP RTT benchmark.

    TCP includes connection overhead (3-way handshake) and guarantees delivery.
    UDP is connectionless with lower overhead but no delivery guarantee.
    This benchmark quantifies the latency difference.
    """

    import statistics

    HOST = "127.0.0.1"
    NUM_MESSAGES = 200  # Reduced for demo speed (production: 1000)
    MSG_SIZE = 100

    def run_tcp_benchmark():
        """Benchmark TCP echo server/client RTT."""
        tcp_port = 19001
        rtts = []

        def tcp_server():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((HOST, tcp_port))
            srv.listen(1)
            conn, _ = srv.accept()
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    # Echo with server timestamp
                    conn.sendall(data)
            except Exception:
                pass
            finally:
                conn.close()
                srv.close()

        server_thread = threading.Thread(target=tcp_server, daemon=True)
        server_thread.start()
        time.sleep(0.3)

        # TCP client
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, tcp_port))

        message = b"X" * MSG_SIZE
        for _ in range(NUM_MESSAGES):
            start = time.perf_counter()
            sock.sendall(message)
            sock.recv(4096)
            end = time.perf_counter()
            rtts.append((end - start) * 1000)  # ms

        sock.close()
        return rtts

    def run_udp_benchmark():
        """Benchmark UDP echo server/client RTT."""
        udp_port = 19002
        rtts = []

        def udp_server():
            srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            srv.bind((HOST, udp_port))
            srv.settimeout(5)
            try:
                for _ in range(NUM_MESSAGES + 10):
                    data, addr = srv.recvfrom(4096)
                    srv.sendto(data, addr)
            except socket.timeout:
                pass
            finally:
                srv.close()

        server_thread = threading.Thread(target=udp_server, daemon=True)
        server_thread.start()
        time.sleep(0.3)

        # UDP client
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)

        message = b"X" * MSG_SIZE
        for _ in range(NUM_MESSAGES):
            start = time.perf_counter()
            sock.sendto(message, (HOST, udp_port))
            try:
                sock.recvfrom(4096)
            except socket.timeout:
                continue
            end = time.perf_counter()
            rtts.append((end - start) * 1000)  # ms

        sock.close()
        return rtts

    print("  TCP vs UDP Protocol Benchmark\n")
    print(f"    Messages: {NUM_MESSAGES}, Payload: {MSG_SIZE} bytes\n")

    # Run benchmarks
    print("    Running TCP benchmark...")
    tcp_rtts = run_tcp_benchmark()

    print("    Running UDP benchmark...")
    udp_rtts = run_udp_benchmark()

    # Compute statistics
    def compute_stats(rtts):
        return {
            "count": len(rtts),
            "min": min(rtts),
            "max": max(rtts),
            "mean": statistics.mean(rtts),
            "std": statistics.stdev(rtts) if len(rtts) > 1 else 0,
        }

    tcp_stats = compute_stats(tcp_rtts)
    udp_stats = compute_stats(udp_rtts)

    # Print comparison table
    print(f"\n    {'Metric':<12} {'TCP':>12} {'UDP':>12}")
    print(f"    {'-'*12} {'-'*12} {'-'*12}")
    print(f"    {'Received':<12} {tcp_stats['count']:>12} {udp_stats['count']:>12}")
    print(f"    {'Min RTT':<12} {tcp_stats['min']:>10.3f}ms {udp_stats['min']:>10.3f}ms")
    print(f"    {'Max RTT':<12} {tcp_stats['max']:>10.3f}ms {udp_stats['max']:>10.3f}ms")
    print(f"    {'Mean RTT':<12} {tcp_stats['mean']:>10.3f}ms {udp_stats['mean']:>10.3f}ms")
    print(f"    {'Std Dev':<12} {tcp_stats['std']:>10.3f}ms {udp_stats['std']:>10.3f}ms")

    # Analysis
    print("\n    --- Analysis ---")
    print(f"""
    UDP typically shows lower mean RTT than TCP because it skips:
    - Connection setup (TCP 3-way handshake adds latency to the first message)
    - Acknowledgment tracking and retransmission logic
    - Nagle's algorithm (TCP may buffer small writes)

    TCP typically shows more consistent latency (lower std dev) because:
    - Guaranteed delivery: no packet loss variation
    - Flow control: prevents congestion-induced jitter

    IoT protocol recommendations:
    - Use TCP (via MQTT) for sensor data where every reading matters
      (e.g., billing meters, safety alerts). Reliability > speed.
    - Use UDP for high-frequency, loss-tolerant streams
      (e.g., real-time video, lidar point clouds). Low latency > reliability.
    - ESP32 benefits more from UDP than Raspberry Pi because its limited
      RAM (520 KB) and CPU (240 MHz) make TCP's overhead proportionally larger.
      The Pi's 1+ GB RAM and GHz CPU absorb TCP overhead with minimal impact.
    """)


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 04: WiFi Networking - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: WiFi Signal Monitor")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: Multi-Client TCP Sensor Server")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: HTTP Sensor Reporter with Retry")
    print("-" * 50)
    exercise_3()

    print("\n\n>>> Exercise 4: Network Device Scanner")
    print("-" * 50)
    exercise_4()

    print("\n\n>>> Exercise 5: TCP vs UDP Benchmark")
    print("-" * 50)
    exercise_5()
