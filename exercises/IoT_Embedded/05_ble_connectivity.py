"""
Exercises for Lesson 05: BLE Connectivity
Topic: IoT_Embedded

Solutions to practice problems from the lesson.
Simulates BLE (Bluetooth Low Energy) operations using Python classes
that mirror the bleak library's async API.

On a real Raspberry Pi with BLE hardware, replace SimulatedBLE* classes
with bleak.BleakScanner and bleak.BleakClient.
"""

import time
import csv
import os
import random
import statistics
from datetime import datetime


# ---------------------------------------------------------------------------
# Simulated BLE layer
# ---------------------------------------------------------------------------

class SimulatedBLEDevice:
    """Simulate a discovered BLE device (mirrors bleak.BLEDevice).

    BLE devices advertise their presence by broadcasting advertisement
    packets on three dedicated channels (37, 38, 39) at 2.4 GHz.
    The RSSI (Received Signal Strength Indicator) in dBm tells us
    how strong the signal is: -30 dBm is very close, -90 dBm is far away.
    """

    def __init__(self, name, address, rssi):
        self.name = name
        self.address = address
        self.rssi = rssi


class SimulatedGATTCharacteristic:
    """Simulate a GATT characteristic."""

    def __init__(self, uuid, properties, handle, value=None):
        self.uuid = uuid
        self.properties = properties
        self.handle = handle
        self._value = value or bytes([0x00])


class SimulatedGATTService:
    """Simulate a GATT service containing characteristics."""

    def __init__(self, uuid, characteristics):
        self.uuid = uuid
        self.characteristics = characteristics


class SimulatedBLEScanner:
    """Simulate bleak.BleakScanner.discover()."""

    _devices = [
        SimulatedBLEDevice("Heart Rate Monitor", "A1:B2:C3:D4:E5:01", -45),
        SimulatedBLEDevice("Env Sensor v2", "A1:B2:C3:D4:E5:02", -62),
        SimulatedBLEDevice("Arduino Nano 33 BLE", "A1:B2:C3:D4:E5:03", -55),
        SimulatedBLEDevice(None, "A1:B2:C3:D4:E5:04", -78),
        SimulatedBLEDevice("Mi Smart Band 5", "A1:B2:C3:D4:E5:05", -68),
    ]

    @classmethod
    def discover(cls, timeout=5.0):
        """Return a list of discovered BLE devices with randomized RSSI."""
        # Simulate RSSI variation between scans (+/- 5 dBm)
        devices = []
        # Sometimes a device disappears (simulates going out of range)
        for d in cls._devices:
            if random.random() > 0.1:
                rssi_variation = random.randint(-5, 5)
                devices.append(SimulatedBLEDevice(
                    d.name, d.address, d.rssi + rssi_variation
                ))
        # Sometimes a new device appears
        if random.random() > 0.6:
            devices.append(SimulatedBLEDevice(
                "Unknown Beacon", f"FF:EE:DD:{random.randint(10,99)}:00:00",
                random.randint(-90, -60)
            ))
        return devices


class SimulatedBLEClient:
    """Simulate bleak.BleakClient for connecting and reading characteristics.

    GATT (Generic Attribute Profile) organizes BLE data into:
    - Services: logical groups (e.g., Heart Rate Service 0x180D)
    - Characteristics: individual data points within a service
    - Properties: what operations are allowed (read, write, notify, indicate)
    """

    # Standard service/characteristic UUIDs
    SERVICES = [
        SimulatedGATTService(
            uuid="0000180d-0000-1000-8000-00805f9b34fb",  # Heart Rate
            characteristics=[
                SimulatedGATTCharacteristic(
                    "00002a37-0000-1000-8000-00805f9b34fb",
                    ["read", "notify"], 0x000e, bytes([0x00, 75])
                ),
            ],
        ),
        SimulatedGATTService(
            uuid="0000180f-0000-1000-8000-00805f9b34fb",  # Battery
            characteristics=[
                SimulatedGATTCharacteristic(
                    "00002a19-0000-1000-8000-00805f9b34fb",
                    ["read"], 0x0012, bytes([85])
                ),
            ],
        ),
        SimulatedGATTService(
            uuid="0000181a-0000-1000-8000-00805f9b34fb",  # Environmental Sensing
            characteristics=[
                SimulatedGATTCharacteristic(
                    "00002a1c-0000-1000-8000-00805f9b34fb",
                    ["read", "notify"], 0x0016, bytes([0xA0, 0x09])  # 24.96 C
                ),
                SimulatedGATTCharacteristic(
                    "00002a6f-0000-1000-8000-00805f9b34fb",
                    ["read", "notify"], 0x001A, bytes([0xE8, 0x13])  # 51.12%
                ),
            ],
        ),
        SimulatedGATTService(
            uuid="0000180a-0000-1000-8000-00805f9b34fb",  # Device Information
            characteristics=[
                SimulatedGATTCharacteristic(
                    "00002a29-0000-1000-8000-00805f9b34fb",
                    ["read"], 0x0020, b"Simulated Mfg"
                ),
                SimulatedGATTCharacteristic(
                    "00002a24-0000-1000-8000-00805f9b34fb",
                    ["read"], 0x0022, b"IoT-Sensor-v1"
                ),
                SimulatedGATTCharacteristic(
                    "00002a26-0000-1000-8000-00805f9b34fb",
                    ["read"], 0x0024, b"1.2.3"
                ),
            ],
        ),
    ]

    def __init__(self, address):
        self.address = address
        self.is_connected = False
        self._notification_callbacks = {}

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    @property
    def services(self):
        return self.SERVICES

    def read_gatt_char(self, uuid):
        """Read a characteristic value by UUID."""
        for service in self.SERVICES:
            for char in service.characteristics:
                if char.uuid == uuid:
                    if "read" not in char.properties:
                        raise Exception(f"Characteristic {uuid} is not readable")
                    return char._value
        raise Exception(f"Characteristic {uuid} not found")

    def start_notify(self, uuid, callback):
        """Subscribe to notifications."""
        self._notification_callbacks[uuid] = callback

    def stop_notify(self, uuid):
        """Unsubscribe from notifications."""
        self._notification_callbacks.pop(uuid, None)

    def simulate_notification(self, uuid, data):
        """Trigger a simulated notification."""
        if uuid in self._notification_callbacks:
            self._notification_callbacks[uuid](uuid, data)


# ---------------------------------------------------------------------------
# Exercise Solutions
# ---------------------------------------------------------------------------

# === Exercise 1: BLE Device Scanner with RSSI Ranking ===
# Problem: Scan BLE devices, rank by RSSI, detect new/disappeared between scans.

def exercise_1():
    """Solution: BLE device scanner with RSSI ranking and change detection."""

    print("  BLE Device Scanner with RSSI Ranking\n")

    previous_devices = {}  # address -> (name, rssi)

    for scan_num in range(1, 4):
        devices = SimulatedBLEScanner.discover(timeout=10.0)

        # Sort by RSSI descending (strongest signal first)
        devices.sort(key=lambda d: d.rssi, reverse=True)

        current_devices = {d.address: (d.name, d.rssi) for d in devices}

        # Detect changes
        current_addrs = set(current_devices.keys())
        previous_addrs = set(previous_devices.keys())
        new_addrs = current_addrs - previous_addrs
        disappeared_addrs = previous_addrs - current_addrs
        persistent_addrs = current_addrs & previous_addrs

        print(f"    --- Scan #{scan_num} (found {len(devices)} devices) ---\n")
        print(f"    {'Rank':<5} {'Name':<25} {'MAC Address':<20} {'RSSI':>6}  {'Change'}")
        print(f"    {'-'*5} {'-'*25} {'-'*20} {'-'*6}  {'-'*15}")

        for rank, device in enumerate(devices, 1):
            name = device.name or "Unknown"
            change = ""
            if device.address in new_addrs:
                change = "** NEW **"
            elif device.address in persistent_addrs:
                old_rssi = previous_devices[device.address][1]
                delta = device.rssi - old_rssi
                if delta != 0:
                    change = f"RSSI {'+' if delta > 0 else ''}{delta} dBm"

            print(f"    {rank:<5} {name:<25} {device.address:<20} {device.rssi:>4} dBm  {change}")

        if disappeared_addrs:
            print(f"\n    Disappeared: {', '.join(sorted(disappeared_addrs))}")

        previous_devices = current_devices
        print()
        time.sleep(0.3)


# === Exercise 2: GATT Service Explorer ===
# Problem: Connect to a BLE device, enumerate all services/characteristics,
# read values, identify standard GATT services.

def exercise_2():
    """Solution: GATT service explorer."""

    STANDARD_SERVICES = {
        "0000180d-0000-1000-8000-00805f9b34fb": "Heart Rate Service",
        "0000180f-0000-1000-8000-00805f9b34fb": "Battery Service",
        "0000181a-0000-1000-8000-00805f9b34fb": "Environmental Sensing Service",
        "0000180a-0000-1000-8000-00805f9b34fb": "Device Information Service",
    }

    STANDARD_CHARS = {
        "00002a37-0000-1000-8000-00805f9b34fb": "Heart Rate Measurement",
        "00002a19-0000-1000-8000-00805f9b34fb": "Battery Level",
        "00002a1c-0000-1000-8000-00805f9b34fb": "Temperature Measurement",
        "00002a6f-0000-1000-8000-00805f9b34fb": "Humidity",
        "00002a29-0000-1000-8000-00805f9b34fb": "Manufacturer Name String",
        "00002a24-0000-1000-8000-00805f9b34fb": "Model Number String",
        "00002a26-0000-1000-8000-00805f9b34fb": "Firmware Revision String",
    }

    print("  GATT Service Explorer\n")

    client = SimulatedBLEClient("A1:B2:C3:D4:E5:02")
    client.connect()
    print(f"    Connected: {client.is_connected}\n")

    for service in client.services:
        service_name = STANDARD_SERVICES.get(service.uuid, "Custom Service")
        print(f"    [Service] {service.uuid}")
        print(f"              Name: {service_name}")

        for char in service.characteristics:
            char_name = STANDARD_CHARS.get(char.uuid, "Custom Characteristic")
            print(f"      [Char] {char.uuid}")
            print(f"              Name: {char_name}")
            print(f"              Properties: {', '.join(char.properties)}")
            print(f"              Handle: 0x{char.handle:04X}")

            if "read" in char.properties:
                try:
                    value = client.read_gatt_char(char.uuid)
                    print(f"              Raw value: {value.hex()}")

                    # Parse known types
                    if char.uuid == "00002a19-0000-1000-8000-00805f9b34fb":
                        print(f"              Parsed: {value[0]}% battery")
                    elif char.uuid == "00002a1c-0000-1000-8000-00805f9b34fb":
                        temp = int.from_bytes(value[:2], "little", signed=True) / 100.0
                        print(f"              Parsed: {temp:.2f} C")
                    elif char.uuid == "00002a6f-0000-1000-8000-00805f9b34fb":
                        hum = int.from_bytes(value[:2], "little") / 100.0
                        print(f"              Parsed: {hum:.2f}%")
                    elif char.uuid in (
                        "00002a29-0000-1000-8000-00805f9b34fb",
                        "00002a24-0000-1000-8000-00805f9b34fb",
                        "00002a26-0000-1000-8000-00805f9b34fb",
                    ):
                        print(f"              Parsed: {value.decode('utf-8')}")
                except Exception as e:
                    print(f"              Read failed: {e}")
        print()

    client.disconnect()


# === Exercise 3: Real-Time Notification Logger ===
# Problem: Subscribe to notifications, log to CSV, parse standard data formats,
# print summary after 60 seconds.

def exercise_3():
    """Solution: BLE notification logger with CSV output and summary stats."""

    csv_file = "/tmp/ble_notifications.csv"
    notifications = []

    def notification_handler(sender_uuid, data):
        """Parse and log BLE notifications.

        Standard GATT data formats:
        - Heart Rate (0x2A37): first byte is flags, second byte is HR value
        - Temperature (0x2A1C): sint16 at 0.01 C resolution
        - Battery (0x2A19): uint8 percentage
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        parsed = None
        if "2a37" in sender_uuid:
            # Heart Rate: byte[1] is the BPM (simplified; flags in byte[0])
            parsed = data[1] if len(data) >= 2 else data[0]
            unit = "bpm"
        elif "2a1c" in sender_uuid:
            # Temperature: signed 16-bit at 0.01 C
            parsed = int.from_bytes(data[:2], "little", signed=True) / 100.0
            unit = "C"
        elif "2a19" in sender_uuid:
            parsed = data[0]
            unit = "%"
        else:
            parsed = int.from_bytes(data, "little")
            unit = "raw"

        entry = {
            "timestamp": ts,
            "uuid": sender_uuid,
            "raw_hex": data.hex(),
            "parsed_value": parsed,
            "unit": unit,
        }
        notifications.append(entry)

    print("  BLE Notification Logger\n")

    client = SimulatedBLEClient("A1:B2:C3:D4:E5:02")
    client.connect()

    # Subscribe to temperature and heart rate notifications
    temp_uuid = "00002a1c-0000-1000-8000-00805f9b34fb"
    hr_uuid = "00002a37-0000-1000-8000-00805f9b34fb"

    client.start_notify(temp_uuid, notification_handler)
    client.start_notify(hr_uuid, notification_handler)
    print("    Subscribed to temperature and heart rate notifications")

    # Initialize CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "characteristic_uuid", "raw_hex", "parsed_value"])

    # Simulate 20 notifications (in real use, these arrive from the BLE device)
    print("    Receiving notifications...\n")
    for i in range(20):
        if random.random() < 0.6:
            # Temperature notification: random value around 24-26 C
            temp_raw = int(random.uniform(2300, 2700))
            data = temp_raw.to_bytes(2, "little", signed=True)
            client.simulate_notification(temp_uuid, data)
        else:
            # Heart rate notification: flags + BPM
            bpm = random.randint(60, 100)
            data = bytes([0x00, bpm])
            client.simulate_notification(hr_uuid, data)

        time.sleep(0.1)

    # Stop notifications
    client.stop_notify(temp_uuid)
    client.stop_notify(hr_uuid)

    # Write all to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        for n in notifications:
            writer.writerow([n["timestamp"], n["uuid"], n["raw_hex"], n["parsed_value"]])

    # Summary
    temp_values = [n["parsed_value"] for n in notifications if n["unit"] == "C"]
    hr_values = [n["parsed_value"] for n in notifications if n["unit"] == "bpm"]

    print(f"    --- Summary ---")
    print(f"    Total notifications: {len(notifications)}")

    if temp_values:
        print(f"\n    Temperature ({len(temp_values)} readings):")
        print(f"      Average: {statistics.mean(temp_values):.2f} C")
        print(f"      Min:     {min(temp_values):.2f} C")
        print(f"      Max:     {max(temp_values):.2f} C")

    if hr_values:
        print(f"\n    Heart Rate ({len(hr_values)} readings):")
        print(f"      Average: {statistics.mean(hr_values):.0f} bpm")
        print(f"      Min:     {min(hr_values)} bpm")
        print(f"      Max:     {max(hr_values)} bpm")

    client.disconnect()

    if os.path.exists(csv_file):
        os.remove(csv_file)


# === Exercise 4: BLE Sensor Monitor with Auto-Reconnect ===
# Problem: Extend BLESensorMonitor with max_retries, exponential backoff,
# connection quality report, and stale notification detection.

def exercise_4():
    """Solution: BLE sensor monitor with production-grade reconnection logic."""

    class RobustBLEMonitor:
        """BLE sensor monitor with exponential backoff and quality tracking.

        Exponential backoff prevents overwhelming a device that is temporarily
        unavailable. Starting at 2s and doubling (2, 4, 8, 16, 32, 60) avoids
        wasting radio bandwidth on futile connection attempts while still
        reconnecting reasonably quickly once the device returns.
        """

        def __init__(self, device_address, max_retries=5):
            self.address = device_address
            self.max_retries = max_retries
            self.client = None

            # Backoff state
            self.backoff = 2  # Start at 2 seconds
            self.max_backoff = 60

            # Quality tracking
            self.connected_at = None
            self.total_uptime = 0.0
            self.total_downtime = 0.0
            self.connection_count = 0
            self.disconnection_count = 0
            self.last_notification_time = None

        def connect(self):
            """Attempt to connect with exponential backoff.

            Returns True on success, raises RuntimeError after max_retries.
            """
            for attempt in range(1, self.max_retries + 1):
                # Simulate 70% connection success rate
                if random.random() < 0.7:
                    self.client = SimulatedBLEClient(self.address)
                    self.client.connect()
                    self.connected_at = datetime.now()
                    self.connection_count += 1
                    self.backoff = 2  # Reset backoff on success
                    print(f"      Connected (attempt {attempt})")
                    return True
                else:
                    print(f"      Connection failed (attempt {attempt}/{self.max_retries}), "
                          f"retrying in {self.backoff}s...")
                    time.sleep(min(self.backoff * 0.1, 0.5))  # Shortened for demo
                    self.backoff = min(self.backoff * 2, self.max_backoff)

            raise RuntimeError(
                f"Failed to connect to {self.address} after {self.max_retries} attempts. "
                f"Check that the device is powered on, in range, and advertising."
            )

        def disconnect(self):
            """Disconnect and update quality metrics."""
            if self.client and self.client.is_connected:
                if self.connected_at:
                    uptime = (datetime.now() - self.connected_at).total_seconds()
                    self.total_uptime += uptime
                self.client.disconnect()
                self.disconnection_count += 1
                self.connected_at = None

        def check_notification_freshness(self, timeout_seconds=30):
            """Warn if no notification received within timeout.

            Stale notifications may indicate the sensor stopped sending data
            even though the BLE connection is technically alive.
            """
            if self.last_notification_time is None:
                return True  # No notifications expected yet

            elapsed = (datetime.now() - self.last_notification_time).total_seconds()
            if elapsed > timeout_seconds:
                print(f"      [WARNING] No notification for {elapsed:.0f}s "
                      f"(threshold: {timeout_seconds}s). Re-subscribing...")
                return False
            return True

        def on_notification(self, sender, data):
            """Handle incoming notification."""
            self.last_notification_time = datetime.now()
            ts = self.last_notification_time.strftime("%H:%M:%S")
            if "2a1c" in sender:
                temp = int.from_bytes(data[:2], "little", signed=True) / 100.0
                print(f"      [{ts}] Temperature: {temp:.2f} C")
            elif "2a6f" in sender:
                hum = int.from_bytes(data[:2], "little") / 100.0
                print(f"      [{ts}] Humidity: {hum:.1f}%")

        def print_quality_report(self):
            """Print connection quality statistics."""
            total_time = self.total_uptime + self.total_downtime
            availability = (self.total_uptime / total_time * 100) if total_time > 0 else 0

            print(f"\n    --- Connection Quality Report ---")
            print(f"      Total uptime:       {self.total_uptime:.1f}s")
            print(f"      Total downtime:     {self.total_downtime:.1f}s")
            print(f"      Availability:       {availability:.1f}%")
            print(f"      Connections:        {self.connection_count}")
            print(f"      Disconnections:     {self.disconnection_count}")
            print(f"      Final backoff:      {self.backoff}s")

    print("  BLE Sensor Monitor with Auto-Reconnect\n")

    monitor = RobustBLEMonitor("A1:B2:C3:D4:E5:02", max_retries=5)
    temp_uuid = "00002a1c-0000-1000-8000-00805f9b34fb"

    # Simulate 2 connection sessions with a disconnection in between
    for session in range(1, 3):
        print(f"\n    --- Session {session} ---")
        try:
            monitor.connect()
            monitor.client.start_notify(temp_uuid, monitor.on_notification)

            # Simulate receiving notifications
            for _ in range(5):
                temp_raw = int(random.uniform(2300, 2700))
                data = temp_raw.to_bytes(2, "little", signed=True)
                monitor.client.simulate_notification(temp_uuid, data)
                time.sleep(0.2)

            # Check freshness
            monitor.check_notification_freshness(timeout_seconds=30)

            monitor.disconnect()
            print(f"      Session {session} ended normally")

        except RuntimeError as e:
            print(f"      ERROR: {e}")
            monitor.total_downtime += 10  # Simulate downtime

    # Simulate a failed reconnection to show RuntimeError
    print("\n    --- Session 3 (forced failure demo) ---")
    bad_monitor = RobustBLEMonitor("FF:FF:FF:FF:FF:FF", max_retries=3)
    # Override simulation to always fail
    original_random = random.random
    random.random = lambda: 0.99  # Always > 0.7, so connect fails
    try:
        bad_monitor.connect()
    except RuntimeError as e:
        print(f"      Caught expected error: {e}")
    finally:
        random.random = original_random

    monitor.print_quality_report()


# === Exercise 5: BLE vs WiFi Trade-off Analysis ===
# Problem: Written analysis comparing BLE and WiFi for three use cases.

def exercise_5():
    """Solution: BLE vs WiFi design trade-off analysis for three IoT scenarios."""

    analysis = """
    === BLE vs WiFi for IoT: Design Trade-off Analysis ===

    --- Scenario 1: Wearable Fitness Tracker (outdoor, 5-minute check-ins) ---

    Recommended: BLE (specifically BLE 5.0 with Coded PHY for outdoor range)

    A wearable fitness tracker has three dominant constraints: battery life,
    physical size, and the need to sync with a smartphone. BLE excels on all
    three. Its idle current draw is 10-50 microamps versus WiFi's milliamps,
    meaning a 100 mAh coin cell can last months rather than hours. Connection
    establishment takes ~6 ms (vs WiFi's multi-second association), so the
    device can wake every 5 minutes, connect, push a few hundred bytes of
    accelerometer/HR data, and sleep -- spending <1 second active per cycle.
    The GATT model's characteristic-based data exchange maps naturally to
    fitness metrics (heart rate characteristic 0x2A37, step count, etc.).
    BLE 5.0 Coded PHY extends outdoor range to 200-400m, covering a running
    track or park. WiFi would require continuous association with a specific
    access point, draining the battery in hours and limiting range to AP coverage.

    --- Scenario 2: Industrial Vibration Sensor (1000 samples/second, continuous) ---

    Recommended: WiFi (802.11n/ac)

    At 1000 samples/second with 16-bit resolution, the sensor generates
    ~16 kbps minimum -- and with headers and timestamps, realistically 50-100 kbps
    sustained. BLE's maximum throughput is ~125-1000 kbps (BLE 5.0 LE 2M PHY),
    but practical application throughput after protocol overhead is 200-400 kbps.
    This leaves almost no headroom for the continuous stream. WiFi provides
    10-100 Mbps, giving comfortable margin for the data rate plus bursts and
    retransmissions. Industrial machines have mains power, eliminating BLE's
    power advantage. TCP over WiFi guarantees no data loss -- critical for
    vibration analysis where a dropped sample corrupts the FFT. The factory
    floor likely already has WiFi infrastructure (access points, DHCP),
    so no new network hardware is needed.

    --- Scenario 3: Smart Door Lock (< 500 ms response, works without WiFi router) ---

    Recommended: BLE (BLE 5.0, LE Secure Connections)

    The 500 ms response requirement is easily met by BLE's ~6 ms connection time
    plus ~10 ms data exchange -- total well under 100 ms. WiFi association alone
    can take 1-3 seconds. Critically, the lock must work when the home WiFi router
    is down (power outage, ISP failure). BLE operates peer-to-peer between the
    phone and the lock with no infrastructure dependency. The GATT write
    operation maps perfectly to a "send unlock command" flow. BLE Secure
    Connections (ECDH + AES-CCM) provides authentication and encryption
    without external certificate infrastructure. Battery operation is
    practical: a BLE lock on two AA batteries can last 6-12 months because
    it spends >99.9% of the time in deep sleep, waking only when it detects
    an advertising scan from an approaching phone. A WiFi lock would need
    to maintain association continuously, consuming 100+ mA and requiring
    frequent battery replacement or permanent wiring.
    """

    print(analysis)


# === Run All Exercises ===
if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 05: BLE Connectivity - Exercise Solutions")
    print("=" * 70)

    print("\n\n>>> Exercise 1: BLE Device Scanner with RSSI Ranking")
    print("-" * 50)
    exercise_1()

    print("\n\n>>> Exercise 2: GATT Service Explorer")
    print("-" * 50)
    exercise_2()

    print("\n\n>>> Exercise 3: Real-Time Notification Logger")
    print("-" * 50)
    exercise_3()

    print("\n\n>>> Exercise 4: BLE Sensor Monitor with Auto-Reconnect")
    print("-" * 50)
    exercise_4()

    print("\n\n>>> Exercise 5: BLE vs WiFi Trade-off Analysis")
    print("-" * 50)
    exercise_5()
