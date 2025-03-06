import asyncio
import struct
import pyautogui

from bleak import BleakClient, BleakScanner


# UUIDs from the Swift program
DISTO_SERVICE_UUID = "3AB10100-F831-4395-B29D-570977D5BF94"
DISTO_DISTANCE_UUID = "3AB10101-F831-4395-B29D-570977D5BF94"
DISTO_COMMAND_UUID = "3AB10109-F831-4395-B29D-570977D5BF94"

# Function to find the DISTO D1 device
async def find_disto():
    devices = await BleakScanner.discover()
    for device in devices:
        if "DISTO" in (device.name or ""):  # Adjust if needed
            print(f"Found Leica DISTO D1: {device.name} - {device.address}")
            return device.address
    print("Leica DISTO D1 not found.")
    return None

# Callback function for receiving distance data
def distance_callback(sender, data):

    #print(f'Data from {sender} received', data)

    if len(data) >= 4:
        distance = struct.unpack('<f', data)[0]
        print(f'Distance: ', distance)

        KEYBORD_WRITE = True
        if KEYBORD_WRITE:
            pyautogui.write(f'{distance*1000:.0f}')  # Types "4.244"
            pyautogui.press('enter')

    #if len(data) >= 4:  # Ensure there are enough bytes
        #distance = int.from_bytes(data[:4], byteorder='little', signed=False) / 1000.0
        #print(f"Distance: {distance:.3f} m")

# Function to connect and read distance
async def read_distance(device_address):
    async with BleakClient(device_address) as client:
        print(f"Connected to {device_address}")

        # Enable notifications for distance data
        await client.start_notify(DISTO_DISTANCE_UUID, distance_callback)

        # Wait for distance data (e.g., 30 seconds)
        await asyncio.sleep(90)

        # Stop notifications
        await client.stop_notify(DISTO_DISTANCE_UUID)

# Main execution
async def main():
    device_address = await find_disto()
    if device_address:
        await read_distance(device_address)

asyncio.run(main())