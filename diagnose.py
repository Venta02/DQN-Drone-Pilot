"""
Diagnostic Script untuk Debug AirSim
=====================================
Script ini membantu mendiagnosa masalah dengan:
1. Distance sensor
2. Drone movement
3. Obstacle detection

Jalankan: python diagnose.py
"""

import airsim
import time
import numpy as np


def check_connection():
    """Test koneksi ke AirSim"""
    print("\n" + "="*60)
    print("1. TESTING KONEKSI AIRSIM")
    print("="*60)
    
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ Berhasil terhubung ke AirSim!")
        return client
    except Exception as e:
        print(f"❌ Gagal terhubung: {e}")
        print("\nPastikan:")
        print("  - AirSim simulator sudah berjalan")
        print("  - Tidak ada aplikasi lain yang menggunakan port 41451")
        return None


def check_distance_sensor(client, drone_name="Drone1"):
    """Test distance sensor"""
    print("\n" + "="*60)
    print("2. TESTING DISTANCE SENSOR")
    print("="*60)
    
    # List semua sensor yang mungkin
    sensor_names = [
        "DistanceSensorFront",
        "DistanceFront", 
        "Distance",
        "DistanceSensor",
        "distance_sensor",
        "front_distance"
    ]
    
    working_sensor = None
    
    for sensor_name in sensor_names:
        try:
            data = client.getDistanceSensorData(
                distance_sensor_name=sensor_name,
                vehicle_name=drone_name
            )
            print(f"\n✅ Sensor '{sensor_name}' DITEMUKAN!")
            print(f"   Distance: {data.distance:.2f} m")
            print(f"   Min Distance: {data.min_distance:.2f} m")
            print(f"   Max Distance: {data.max_distance:.2f} m")
            working_sensor = sensor_name
            
            if data.distance >= data.max_distance * 0.99:
                print(f"\n   ⚠️ WARNING: Jarak = MaxDistance ({data.max_distance}m)")
                print(f"   Ini berarti TIDAK ADA obstacle terdeteksi di depan drone!")
                print(f"   Pastikan ada obstacle di dalam range sensor.")
            
        except Exception as e:
            print(f"❌ Sensor '{sensor_name}': tidak tersedia")
    
    if working_sensor is None:
        print("\n❌ TIDAK ADA distance sensor yang terdeteksi!")
        print("\nSolusi:")
        print("  1. Edit file settings.json AirSim (Documents/AirSim/settings.json)")
        print("  2. Tambahkan konfigurasi sensor di bagian 'Sensors'")
        print("  3. Restart AirSim")
        print("\nContoh konfigurasi sensor:")
        print('''
{
  "Vehicles": {
    "Drone1": {
      "Sensors": {
        "DistanceSensorFront": {
          "SensorType": 5,
          "Enabled": true,
          "MinDistance": 0.2,
          "MaxDistance": 50,
          "X": 0.5, "Y": 0, "Z": 0,
          "Yaw": 0, "Pitch": 0, "Roll": 0
        }
      }
    }
  }
}
        ''')
    
    return working_sensor


def check_lidar(client, drone_name="Drone1"):
    """Test LiDAR sebagai fallback"""
    print("\n" + "="*60)
    print("3. TESTING LIDAR (FALLBACK)")
    print("="*60)
    
    try:
        lidar_data = client.getLidarData(vehicle_name=drone_name)
        points = len(lidar_data.point_cloud) // 3
        
        if points > 0:
            print(f"✅ LiDAR aktif! Points: {points}")
            point_array = np.array(lidar_data.point_cloud).reshape(-1, 3)
            front_points = point_array[point_array[:, 0] > 0]
            if len(front_points) > 0:
                min_dist = np.min(np.linalg.norm(front_points, axis=1))
                print(f"   Jarak terdekat di depan: {min_dist:.2f} m")
        else:
            print("⚠️ LiDAR aktif tapi tidak ada point cloud")
            print("   Mungkin tidak ada objek dalam jangkauan")
    except Exception as e:
        print(f"❌ LiDAR tidak tersedia: {e}")


def test_drone_movement(client, drone_name="Drone1"):
    """Test pergerakan drone"""
    print("\n" + "="*60)
    print("4. TESTING DRONE MOVEMENT")
    print("="*60)
    
    try:
        # Enable API control
        client.enableApiControl(True, drone_name)
        client.armDisarm(True, drone_name)
        
        # Get initial position
        state = client.getMultirotorState(drone_name)
        pos_before = state.kinematics_estimated.position
        print(f"\nPosisi awal: x={pos_before.x_val:.2f}, y={pos_before.y_val:.2f}, z={pos_before.z_val:.2f}")
        
        # Takeoff
        print("\n🚁 Taking off...")
        client.takeoffAsync(vehicle_name=drone_name).join()
        time.sleep(2)
        
        # Move forward
        print("🚁 Moving forward (5 m/s for 2 seconds)...")
        client.moveByVelocityAsync(5, 0, 0, 2, vehicle_name=drone_name).join()
        time.sleep(1)
        
        # Get new position
        state = client.getMultirotorState(drone_name)
        pos_after = state.kinematics_estimated.position
        print(f"Posisi setelah: x={pos_after.x_val:.2f}, y={pos_after.y_val:.2f}, z={pos_after.z_val:.2f}")
        
        # Calculate distance moved
        distance_moved = np.sqrt(
            (pos_after.x_val - pos_before.x_val)**2 +
            (pos_after.y_val - pos_before.y_val)**2
        )
        print(f"\n📏 Jarak yang ditempuh: {distance_moved:.2f} m")
        
        if distance_moved > 1:
            print("✅ Drone bergerak dengan baik!")
        else:
            print("⚠️ Drone tidak bergerak banyak. Cek apakah ada collision/obstacle.")
        
        # Stop and land
        print("\n🚁 Stopping and landing...")
        client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=drone_name).join()
        client.landAsync(vehicle_name=drone_name).join()
        
    except Exception as e:
        print(f"❌ Error saat testing movement: {e}")
    finally:
        client.armDisarm(False, drone_name)
        client.enableApiControl(False, drone_name)


def continuous_distance_monitor(client, drone_name="Drone1", duration=10):
    """Monitor jarak secara kontinyu"""
    print("\n" + "="*60)
    print("5. MONITORING JARAK (REAL-TIME)")
    print("="*60)
    print(f"Monitoring selama {duration} detik... (Gerakkan drone atau objek)")
    print("Tekan Ctrl+C untuk stop\n")
    
    try:
        client.enableApiControl(True, drone_name)
        client.armDisarm(True, drone_name)
        client.takeoffAsync(vehicle_name=drone_name).join()
        time.sleep(1)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                data = client.getDistanceSensorData(
                    distance_sensor_name="DistanceSensorFront",
                    vehicle_name=drone_name
                )
                distance = data.distance
                
                # Visual bar
                bar_length = min(int(distance / 2), 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                obstacle_status = "NO OBSTACLE" if distance >= data.max_distance * 0.9 else "DETECTED!"
                print(f"\rDistance: {distance:7.2f}m [{bar}] {obstacle_status}    ", end="")
                
            except:
                print("\r[Sensor error]                                              ", end="")
            
            time.sleep(0.1)
        
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\nMonitoring dihentikan.")
    finally:
        client.landAsync(vehicle_name=drone_name).join()
        client.armDisarm(False, drone_name)
        client.enableApiControl(False, drone_name)


def check_obstacles_in_scene(client):
    """Cek apakah ada obstacle di scene"""
    print("\n" + "="*60)
    print("6. CEK OBSTACLE DI SCENE")
    print("="*60)
    
    print("""
⚠️ PENTING: Pastikan ada OBSTACLE di depan drone di AirSim!

Jika menggunakan environment default AirSim:
- Neighborhood: Ada rumah dan bangunan
- Blocks: Ada blok-blok besar
- LandscapeMountains: Terrain terbuka (mungkin tidak ada obstacle dekat)

Solusi jika tidak ada obstacle:
1. Gunakan environment dengan bangunan (Neighborhood/Blocks)
2. Atau tambahkan obstacle manual di Unreal Editor
3. Posisikan drone di dekat dinding/bangunan

Posisi drone saat ini:
""")
    
    try:
        state = client.getMultirotorState("Drone1")
        pos = state.kinematics_estimated.position
        print(f"   X: {pos.x_val:.2f} m")
        print(f"   Y: {pos.y_val:.2f} m")
        print(f"   Z: {pos.z_val:.2f} m (negatif = di atas ground)")
    except:
        print("   [Tidak bisa membaca posisi]")


def main():
    print("\n" + "="*60)
    print("     AIRSIM DRONE DIAGNOSTIC TOOL")
    print("="*60)
    
    # Step 1: Check connection
    client = check_connection()
    if client is None:
        return
    
    # Step 2: Check distance sensor
    sensor = check_distance_sensor(client)
    
    # Step 3: Check LiDAR
    check_lidar(client)
    
    # Step 4: Check obstacles
    check_obstacles_in_scene(client)
    
    # Ask user what to do next
    print("\n" + "="*60)
    print("PILIH TEST LANJUTAN:")
    print("="*60)
    print("1. Test drone movement")
    print("2. Monitor jarak real-time (10 detik)")
    print("3. Skip")
    
    choice = input("\nPilihan (1/2/3): ").strip()
    
    if choice == "1":
        test_drone_movement(client)
    elif choice == "2":
        continuous_distance_monitor(client)
    
    print("\n" + "="*60)
    print("DIAGNOSIS SELESAI")
    print("="*60)
    print("""
KESIMPULAN & REKOMENDASI:

1. Jika distance sensor TIDAK terdeteksi:
   → Update settings.json AirSim dengan konfigurasi sensor
   → Restart AirSim

2. Jika distance = MaxDistance (2500m atau 50m):
   → Tidak ada obstacle di depan drone
   → Pindahkan drone ke dekat dinding/bangunan
   → Atau gunakan environment dengan bangunan

3. Jika drone TIDAK bergerak:
   → Cek collision dengan ground/objek
   → Pastikan takeoff berhasil
   → Cek ketinggian initial (z harus negatif, misal -5)

4. Jika semua OK tapi training tetap gagal:
   → Pastikan ada obstacle dalam jarak 50m di depan drone
   → Sesuaikan initial_position di config.py
""")


if __name__ == "__main__":
    main()
