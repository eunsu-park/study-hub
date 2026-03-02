#!/usr/bin/env python3
"""
BLE (Bluetooth Low Energy) Device Scanner and GATT Client Example

This script provides the following features:
1. BLE device scanning (simulation mode supported)
2. GATT service and characteristic discovery
3. BLE characteristic value reading
4. BLE notification reception
5. Sensor data reception example

Reference: content/ko/IoT_Embedded/05_BLE_Connectivity.md

Note: The bleak library is required for actual BLE functionality.
      pip install bleak
"""

import asyncio
import sys
import time
import random
import struct
from typing import Optional, List, Dict, Callable
from datetime import datetime


# ============================================================================
# BLE Library Import (Optional)
# ============================================================================

# Why: Graceful import fallback lets this script run as a teaching tool on any
# machine. Students without Bluetooth hardware still see realistic output
# through the simulation path.
try:
    from bleak import BleakScanner, BleakClient
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    print("Warning: bleak library not found.")
    print("   Running in simulation mode.")
    print("   To use actual BLE features, run 'pip install bleak'.\n")


# ============================================================================
# Standard BLE UUIDs
# ============================================================================

class BLE_UUID:
    """Standard BLE Service and Characteristic UUIDs"""

    # Standard service UUIDs (16-bit)
    GENERIC_ACCESS = "00001800-0000-1000-8000-00805f9b34fb"
    GENERIC_ATTRIBUTE = "00001801-0000-1000-8000-00805f9b34fb"
    DEVICE_INFORMATION = "0000180a-0000-1000-8000-00805f9b34fb"
    BATTERY_SERVICE = "0000180f-0000-1000-8000-00805f9b34fb"
    ENVIRONMENTAL_SENSING = "0000181a-0000-1000-8000-00805f9b34fb"
    HEART_RATE = "0000180d-0000-1000-8000-00805f9b34fb"

    # Standard characteristic UUIDs
    DEVICE_NAME = "00002a00-0000-1000-8000-00805f9b34fb"
    BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
    TEMPERATURE = "00002a6e-0000-1000-8000-00805f9b34fb"
    HUMIDITY = "00002a6f-0000-1000-8000-00805f9b34fb"
    HEART_RATE_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"

    # Why: The BLE spec defines a "Bluetooth Base UUID" so that standard 16-bit
    # service IDs can be represented as full 128-bit UUIDs. This conversion is
    # needed when the BLE library reports only 128-bit UUIDs.
    @staticmethod
    def uuid_16_to_128(uuid_16: str) -> str:
        """
        Convert 16-bit UUID to 128-bit BLE Base UUID

        Args:
            uuid_16: 16-bit UUID (e.g., "0x180F")

        Returns:
            str: 128-bit UUID
        """
        base_uuid = "00000000-0000-1000-8000-00805f9b34fb"
        uuid_16_clean = uuid_16.replace("0x", "").lower()
        return f"0000{uuid_16_clean}{base_uuid[8:]}"


# ============================================================================
# Simulation Mode
# ============================================================================

class SimulatedBLEDevice:
    """Simulated BLE Device"""

    def __init__(self, name: str, address: str, rssi: int):
        self.name = name
        self.address = address
        self.rssi = rssi

    def __repr__(self):
        return f"SimulatedBLEDevice(name='{self.name}', address='{self.address}', rssi={self.rssi})"


def simulate_ble_scan(timeout: float = 10.0) -> List[SimulatedBLEDevice]:
    """
    BLE Scan Simulation

    Args:
        timeout: Scan duration (seconds)

    Returns:
        list: List of simulated BLE devices
    """
    print(f"[Simulation] Scanning for BLE devices... ({timeout} seconds)")
    time.sleep(2)  # Simulate scan

    devices = [
        SimulatedBLEDevice("TempSensor-01", "AA:BB:CC:DD:EE:01", -45),
        SimulatedBLEDevice("HeartRate-BLE", "AA:BB:CC:DD:EE:02", -52),
        SimulatedBLEDevice("Battery-Monitor", "AA:BB:CC:DD:EE:03", -38),
        SimulatedBLEDevice("EnvSensor", "AA:BB:CC:DD:EE:04", -61),
        SimulatedBLEDevice("Smart-Watch", "AA:BB:CC:DD:EE:05", -55),
        SimulatedBLEDevice(None, "AA:BB:CC:DD:EE:06", -72),  # Unnamed device
    ]

    return devices


# Why: Returning raw bytes (not Python floats) mimics the real BLE GATT protocol,
# where each characteristic packs data into a fixed binary format defined by the
# Bluetooth SIG. This teaches students to use struct.unpack for BLE payloads.
def simulate_read_characteristic(char_uuid: str) -> bytes:
    """
    BLE Characteristic Read Simulation

    Args:
        char_uuid: Characteristic UUID

    Returns:
        bytes: Simulated data
    """
    if BLE_UUID.BATTERY_LEVEL in char_uuid:
        # Battery level (0-100%)
        return bytes([random.randint(50, 100)])

    elif BLE_UUID.TEMPERATURE in char_uuid:
        # Temperature (0.01 degree units, 16-bit integer)
        temp = random.uniform(20.0, 30.0)
        temp_raw = int(temp * 100)
        return struct.pack('<h', temp_raw)

    elif BLE_UUID.HUMIDITY in char_uuid:
        # Humidity (0.01% units, 16-bit unsigned integer)
        humidity = random.uniform(40.0, 70.0)
        humidity_raw = int(humidity * 100)
        return struct.pack('<H', humidity_raw)

    else:
        # Default value
        return b'\x00\x00'


# ============================================================================
# BLE Scan Functions
# ============================================================================

async def scan_ble_devices(timeout: float = 10.0, use_simulation: bool = False) -> List:
    """
    BLE Device Scan

    Args:
        timeout: Scan duration (seconds)
        use_simulation: Force simulation mode

    Returns:
        list: List of discovered BLE devices
    """
    if not BLEAK_AVAILABLE or use_simulation:
        return simulate_ble_scan(timeout)

    print(f"Scanning for BLE devices... ({timeout} seconds)")

    try:
        devices = await BleakScanner.discover(timeout=timeout)
        return devices
    except Exception as e:
        print(f"Scan error: {e}")
        print("Switching to simulation mode...")
        return simulate_ble_scan(timeout)


async def scan_with_filter(name_filter: Optional[str] = None, timeout: float = 10.0) -> List:
    """
    Filtered BLE Scan

    Args:
        name_filter: Device name filter (partial match)
        timeout: Scan duration

    Returns:
        list: Filtered device list
    """
    devices = await scan_ble_devices(timeout)

    if name_filter:
        devices = [d for d in devices if d.name and name_filter.lower() in d.name.lower()]

    return devices


async def continuous_scan(duration: float = 30.0, callback: Optional[Callable] = None):
    """
    Continuous BLE Scan

    Args:
        duration: Scan duration (seconds)
        callback: Callback function when device is discovered
    """
    if not BLEAK_AVAILABLE:
        print("[Simulation] Continuous scan is not supported in simulation mode.")
        devices = simulate_ble_scan(duration)
        for device in devices:
            print(f"Discovered: {device.name} ({device.address}) - RSSI: {device.rssi} dBm")
        return

    def detection_callback(device, advertisement_data):
        print(f"Discovered: {device.name or 'Unknown'} ({device.address}) - RSSI: {device.rssi} dBm")
        if callback:
            callback(device, advertisement_data)

    scanner = BleakScanner(detection_callback=detection_callback)

    print(f"Starting continuous scan ({duration} seconds)")
    await scanner.start()
    await asyncio.sleep(duration)
    await scanner.stop()
    print("Scan finished")


# ============================================================================
# BLE Connection and Discovery
# ============================================================================

async def connect_and_explore(address: str, use_simulation: bool = False):
    """
    BLE Device Connection and Service/Characteristic Discovery

    Args:
        address: BLE device MAC address
        use_simulation: Use simulation mode
    """
    if not BLEAK_AVAILABLE or use_simulation:
        print(f"[Simulation] Connecting: {address}")
        print(f"[Simulation] Connected!")
        print(f"\nService: {BLE_UUID.ENVIRONMENTAL_SENSING}")
        print(f"  Description: Environmental Sensing")
        print(f"    Characteristic: {BLE_UUID.TEMPERATURE}")
        print(f"      Properties: ['read', 'notify']")
        print(f"      Value: {simulate_read_characteristic(BLE_UUID.TEMPERATURE).hex()}")
        print(f"    Characteristic: {BLE_UUID.HUMIDITY}")
        print(f"      Properties: ['read', 'notify']")
        print(f"      Value: {simulate_read_characteristic(BLE_UUID.HUMIDITY).hex()}")
        return

    print(f"Connecting: {address}")

    try:
        async with BleakClient(address) as client:
            print(f"Connected! MTU: {client.mtu_size}")

            # Service discovery
            for service in client.services:
                print(f"\nService: {service.uuid}")
                print(f"  Description: {service.description}")

                # Characteristic discovery
                for char in service.characteristics:
                    print(f"    Characteristic: {char.uuid}")
                    print(f"      Properties: {char.properties}")

                    # Read value if readable
                    if "read" in char.properties:
                        try:
                            value = await client.read_gatt_char(char.uuid)
                            print(f"      Value: {value.hex()}")
                        except Exception as e:
                            print(f"      Read failed: {e}")

    except Exception as e:
        print(f"Connection error: {e}")
        print("Retrying in simulation mode...")
        await connect_and_explore(address, use_simulation=True)


# ============================================================================
# BLE Sensor Data Reading
# ============================================================================

async def read_sensor_data(address: str, use_simulation: bool = False) -> Dict:
    """
    BLE Sensor Data Reading

    Args:
        address: BLE device address
        use_simulation: Simulation mode

    Returns:
        dict: Sensor data
    """
    result = {}

    if not BLEAK_AVAILABLE or use_simulation:
        print(f"[Simulation] Reading sensor data: {address}")

        # Battery
        battery_data = simulate_read_characteristic(BLE_UUID.BATTERY_LEVEL)
        result['battery'] = battery_data[0]

        # Temperature
        temp_data = simulate_read_characteristic(BLE_UUID.TEMPERATURE)
        temp_raw = struct.unpack('<h', temp_data)[0]
        result['temperature'] = temp_raw * 0.01

        # Humidity
        humidity_data = simulate_read_characteristic(BLE_UUID.HUMIDITY)
        humidity_raw = struct.unpack('<H', humidity_data)[0]
        result['humidity'] = humidity_raw * 0.01

        return result

    try:
        async with BleakClient(address) as client:
            # Battery level
            try:
                data = await client.read_gatt_char(BLE_UUID.BATTERY_LEVEL)
                result['battery'] = data[0]
            except Exception:
                pass

            # Temperature
            try:
                data = await client.read_gatt_char(BLE_UUID.TEMPERATURE)
                temp_raw = struct.unpack('<h', data[:2])[0]
                result['temperature'] = temp_raw * 0.01
            except Exception:
                pass

            # Humidity
            try:
                data = await client.read_gatt_char(BLE_UUID.HUMIDITY)
                humidity_raw = struct.unpack('<H', data[:2])[0]
                result['humidity'] = humidity_raw * 0.01
            except Exception:
                pass

    except Exception as e:
        print(f"Read error: {e}")
        print("Retrying in simulation mode...")
        return await read_sensor_data(address, use_simulation=True)

    return result


# ============================================================================
# BLE Notification Reception
# ============================================================================

# Why: A factory function is used because bleak's start_notify() expects a
# callback with a fixed signature (sender, data). The closure captures the
# sensor_type, letting one factory produce type-specific decoders.
def create_notification_handler(sensor_type: str):
    """
    Create Notification Handler

    Args:
        sensor_type: Sensor type ('temperature', 'humidity', etc.)

    Returns:
        function: Notification handler function
    """
    def handler(sender, data):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Received ({sensor_type}): {data.hex()}")

        if sensor_type == 'temperature':
            temp_raw = struct.unpack('<h', data[:2])[0]
            temp = temp_raw * 0.01
            print(f"  Temperature: {temp:.2f}C")

        elif sensor_type == 'humidity':
            humidity_raw = struct.unpack('<H', data[:2])[0]
            humidity = humidity_raw * 0.01
            print(f"  Humidity: {humidity:.2f}%")

        elif sensor_type == 'battery':
            battery = data[0]
            print(f"  Battery: {battery}%")

        elif sensor_type == 'heart_rate':
            flags = data[0]
            if flags & 0x01:  # 16-bit heart rate
                hr = int.from_bytes(data[1:3], 'little')
            else:  # 8-bit heart rate
                hr = data[1]
            print(f"  Heart rate: {hr} bpm")

    return handler


async def subscribe_notifications(
    address: str,
    char_uuid: str,
    sensor_type: str = 'unknown',
    duration: float = 60,
    use_simulation: bool = False
):
    """
    BLE Notification Subscription

    Args:
        address: BLE device address
        char_uuid: Characteristic UUID
        sensor_type: Sensor type
        duration: Subscription duration (seconds)
        use_simulation: Simulation mode
    """
    if not BLEAK_AVAILABLE or use_simulation:
        print(f"[Simulation] Notification subscription: {address}")
        print(f"[Simulation] Characteristic: {char_uuid}")
        print(f"\nReceiving simulation data for {duration} seconds...\n")

        handler = create_notification_handler(sensor_type)

        for i in range(int(duration / 2)):
            # Generate simulation data
            data = simulate_read_characteristic(char_uuid)
            handler(char_uuid, data)
            await asyncio.sleep(2)

        print("\nSubscription ended")
        return

    print(f"Connecting: {address}")

    try:
        async with BleakClient(address) as client:
            print(f"Connected!")

            handler = create_notification_handler(sensor_type)

            # Start notifications
            await client.start_notify(char_uuid, handler)
            print(f"Notification subscription started: {char_uuid}")
            print(f"Receiving for {duration} seconds...\n")

            # Receive for specified duration
            await asyncio.sleep(duration)

            # Stop notifications
            await client.stop_notify(char_uuid)
            print("\nNotification subscription ended")

    except Exception as e:
        print(f"Subscription error: {e}")
        print("Retrying in simulation mode...")
        await subscribe_notifications(address, char_uuid, sensor_type, duration, use_simulation=True)


# ============================================================================
# BLE Sensor Monitor Class
# ============================================================================

class BLESensorMonitor:
    """BLE Environmental Sensor Monitoring Class"""

    def __init__(self, device_address: Optional[str] = None, use_simulation: bool = False):
        self.device_address = device_address
        self.use_simulation = use_simulation or not BLEAK_AVAILABLE
        self.data_buffer = []

    async def start_monitoring(self, duration: float = 60):
        """
        Start Monitoring

        Args:
            duration: Monitoring duration (seconds)
        """
        if not self.device_address:
            print("Error: Device address not specified.")
            return

        print(f"=== BLE Sensor Monitoring Started ===")
        print(f"Device: {self.device_address}")
        print(f"Duration: {duration} seconds")
        print(f"Mode: {'Simulation' if self.use_simulation else 'Real'}\n")

        if self.use_simulation:
            # Simulation monitoring
            for i in range(int(duration / 5)):
                data = await read_sensor_data(self.device_address, use_simulation=True)
                timestamp = datetime.now()

                print(f"[{timestamp.strftime('%H:%M:%S')}] Received:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
                    self.data_buffer.append({
                        'type': key,
                        'value': value,
                        'timestamp': timestamp
                    })

                await asyncio.sleep(5)
        else:
            # Real BLE monitoring
            await subscribe_notifications(
                self.device_address,
                BLE_UUID.TEMPERATURE,
                'temperature',
                duration / 2
            )

        print("\n=== Monitoring Ended ===")
        print(f"Collected data points: {len(self.data_buffer)}")

    def get_summary(self) -> Dict:
        """Collected data summary"""
        if not self.data_buffer:
            return {}

        summary = {}
        data_types = set(d['type'] for d in self.data_buffer)

        for dtype in data_types:
            values = [d['value'] for d in self.data_buffer if d['type'] == dtype]
            summary[dtype] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values)
            }

        return summary


# ============================================================================
# Main Function
# ============================================================================

def print_devices(devices: List):
    """Print device list"""
    print(f"\nDiscovered devices: {len(devices)}\n")

    for i, device in enumerate(devices, 1):
        name = device.name or 'Unknown'
        address = device.address
        rssi = getattr(device, 'rssi', 'N/A')
        print(f"{i:2}. {name:20} - {address} (RSSI: {rssi} dBm)")


async def main_async():
    """Async main function"""
    if len(sys.argv) < 2:
        print("BLE Device Scanner and GATT Client Example")
        print("\nUsage:")
        print("  python ble_scanner.py scan              - Scan for BLE devices")
        print("  python ble_scanner.py scan <filter>      - Scan with name filter")
        print("  python ble_scanner.py explore <address>   - Explore device")
        print("  python ble_scanner.py read <address>      - Read sensor data")
        print("  python ble_scanner.py notify <address>    - Receive notifications")
        print("  python ble_scanner.py monitor <address>   - Monitor sensors")
        print("\nExamples:")
        print("  python ble_scanner.py scan")
        print("  python ble_scanner.py scan temp")
        print("  python ble_scanner.py explore AA:BB:CC:DD:EE:FF")
        print("\nNote: Runs in simulation mode if bleak library is not installed.")
        return

    command = sys.argv[1].lower()

    if command == 'scan':
        name_filter = sys.argv[2] if len(sys.argv) > 2 else None
        if name_filter:
            devices = await scan_with_filter(name_filter, timeout=10.0)
        else:
            devices = await scan_ble_devices(timeout=10.0)
        print_devices(devices)

    elif command == 'explore':
        if len(sys.argv) < 3:
            print("Error: Please enter a device address")
            print("Example: python ble_scanner.py explore AA:BB:CC:DD:EE:FF")
            return
        address = sys.argv[2]
        await connect_and_explore(address)

    elif command == 'read':
        if len(sys.argv) < 3:
            print("Error: Please enter a device address")
            return
        address = sys.argv[2]
        data = await read_sensor_data(address)
        print("\n=== Sensor Data ===")
        for key, value in data.items():
            print(f"  {key}: {value}")

    elif command == 'notify':
        if len(sys.argv) < 3:
            print("Error: Please enter a device address")
            return
        address = sys.argv[2]
        await subscribe_notifications(
            address,
            BLE_UUID.TEMPERATURE,
            'temperature',
            duration=30
        )

    elif command == 'monitor':
        if len(sys.argv) < 3:
            print("Error: Please enter a device address")
            return
        address = sys.argv[2]
        monitor = BLESensorMonitor(address)
        await monitor.start_monitoring(duration=30)

        # Print summary
        summary = monitor.get_summary()
        if summary:
            print("\n=== Data Summary ===")
            for sensor, stats in summary.items():
                print(f"{sensor}:")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
                print(f"  Avg: {stats['avg']:.2f}")
                print(f"  Count: {stats['count']}")

    else:
        print(f"Unknown command: {command}")
        print("Run 'python ble_scanner.py' for help")


def main():
    """Main function"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")


if __name__ == "__main__":
    main()
