"""
3D Character Game with MPU6050 Control
Features:
- 3D character that moves based on MPU6050 tilt data
- Simple 3D world with obstacles
- Auto-connects to Bluetooth MPU6050 controller
"""

import sys
import math
import time
import threading
import serial
import serial.tools.list_ports
import numpy as np
from collections import deque

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pygame PyOpenGL PyOpenGL-accelerate pyserial numpy")
    sys.exit(1)

# ---------------------- Bluetooth Controller ----------------------
class BluetoothController:
    def __init__(self, baudrate=38400, timeout=0.1, alpha=0.3):
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.connected = False
        self.running = False
        
        # IMU state and smoothing
        self.raw_data = {'accel': {}, 'gyro': {}}
        self.smoothed = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.data_history = deque(maxlen=50)

        # complementary filter & integration state
        self._alpha = alpha
        self._lock = threading.Lock()
        self._last_time = None
        # yaw is best integrated from gyro (degrees)
        self._yaw = 0.0

    def scan_ports(self):
        """Print available serial ports"""
        print("\n" + "="*60)
        print("Available Serial Ports:")
        print("="*60)
        
        ports = list(serial.tools.list_ports.comports())
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device} - {port.description}")
            if "bluetooth" in port.description.lower():
                print("      ^ Bluetooth device detected!")
        print()
        return ports

    def connect(self, port_hint=None):
        """Connect to Bluetooth device"""
        # First try the port hint if provided
        if port_hint:
            print(f"Trying user-specified port {port_hint}...")
            try:
                self.serial_conn = serial.Serial(port_hint, 
                                              baudrate=self.baudrate,
                                              timeout=self.timeout)
                # Test if we can read data
                time.sleep(2)  # Give time for the device to initialize
                self.serial_conn.reset_input_buffer()
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if '|' in line and 'Gyro' in line:
                    print(f"✅ Connected to {port_hint}")
                    print(f"Initial data: {line[:100]}...")
                    self.connected = True
                    return True
                else:
                    self.serial_conn.close()
            except Exception as e:
                print(f"✗ Failed to connect to {port_hint}: {e}")
        
        # If port_hint failed or not provided, scan all ports
        print("\nScanning all available ports for MPU6050 data...")
        ports = [p.device for p in serial.tools.list_ports.comports()]
        
        for port in ports:
            try:
                print(f"\nTrying {port}...")
                ser = serial.Serial(port, baudrate=self.baudrate, timeout=1)
                print(f"  - Port opened, waiting for data...")
                
                # Wait a bit for data to start coming in
                time.sleep(2)
                ser.reset_input_buffer()
                
                # Read a line of data
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"  - Received: {line[:100]}")
                
                # Check if this looks like our MPU6050 data
                if '|' in line and 'Gyro' in line:
                    print(f"✅ Found MPU6050 on {port}")
                    print(f"   Sample data: {line[:100]}...")
                    self.serial_conn = ser
                    self.connected = True
                    return True
                else:
                    print(f"  - Not MPU6050 data, trying next port...")
                    ser.close()
                    
            except Exception as e:
                print(f"  - Error on {port}: {str(e)[:50]}...")
                continue
                
        print("\n⚠️  No working Bluetooth connection with MPU6050 data found")
        print("Please check:")
        print("1. The HC-05 is powered and paired with your computer")
        print("2. The correct baud rate is set (38400)")
        print("3. The Arduino is running and sending data")
        self.connected = False
        return False

    def start(self):
        """Start listening for data"""
        if not self.connected:
            return False
        self.running = True
        self.data_thread = threading.Thread(target=self._data_loop, daemon=True)
        self.data_thread.start()
        return True

    def stop(self):
        """Stop listening and close connection"""
        self.running = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=0.5)
        if self.serial_conn:
            self.serial_conn.close()
        self.connected = False

    def _data_loop(self):
        """Main data reading loop - optimized for minimal latency"""
        buffer = ""
        while self.running and self.serial_conn:
            try:
                # Read all available bytes without blocking
                if self.serial_conn.in_waiting > 0:
                    buffer += self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        # Parse flexible IMU formats such as:
                        # "Y:-0.02 Z:-0.39 | Gyro => X:-35.55 Y:7.18 Z:-3.84"
                        # or "Accel => X:0.01 Y:-0.02 Z:0.99 | Gyro => X:0.1 Y:-0.2 Z:1.5"
                        try:
                            if '|' not in line:
                                continue
                            accel_part, gyro_part = line.split('|', 1)

                            # parse a part like "Y:-0.02 Z:-0.39" or "Accel => X:.. Y:.. Z:.."
                            def parse_kv(text):
                                out = {}
                                # strip labels
                                text = text.replace('Accel', '').replace('Gyro', '').replace('=>', '')
                                for tok in text.strip().split():
                                    if ':' in tok:
                                        k, v = tok.split(':', 1)
                                        try:
                                            out[k.strip()] = float(v)
                                        except ValueError:
                                            continue
                                return out

                            accel_data = parse_kv(accel_part)
                            gyro_data = parse_kv(gyro_part)

                            # compute dt
                            now = time.time()
                            if self._last_time is None:
                                dt = 0.02
                            else:
                                dt = max(0.001, now - self._last_time)
                            self._last_time = now

                            # Pass parsed data to processor (thread-safe)
                            self._process_imu(accel_data, gyro_data, dt)

                        except Exception:
                            # swallow parse errors and continue
                            continue
                else:
                    # Small sleep to prevent 100% CPU usage when no data is available
                    time.sleep(0.001)
                
            except Exception as e:
                print(f"Data read error: {e}")
                time.sleep(0.1)  # Small delay on error to prevent busy loop

    def _process_imu(self, accel_data, gyro_data, dt):
        """Process accelerometer and gyroscope readings into orientation and motion.

        accel_data: dict with keys like 'X','Y','Z' (in g)
        gyro_data: dict with keys like 'X','Y','Z' (deg/s or raw - we treat as deg/s)
        dt: seconds since last sample
        """
        with self._lock:
            # store raw
            self.raw_data['accel'] = accel_data.copy()
            self.raw_data['gyro'] = gyro_data.copy()

            # compute pitch and roll from accelerometer (radians)
            ax = accel_data.get('X', accel_data.get('x', 0.0))
            ay = accel_data.get('Y', accel_data.get('y', 0.0))
            az = accel_data.get('Z', accel_data.get('z', 0.0))

            # Avoid div-by-zero
            try:
                roll_acc = math.degrees(math.atan2(ay, az))
                pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
            except Exception:
                roll_acc = 0.0
                pitch_acc = 0.0

            # integrate gyro for yaw (use Z axis); accept either uppercase or lowercase keys
            gz = gyro_data.get('Z', gyro_data.get('z', 0.0))
            # Assume gyro given in deg/s; integrate
            self._yaw += gz * dt

            # complementary filter between accel and previous smoothed values for pitch/roll
            alpha = self._alpha
            self.smoothed['pitch'] = pitch_acc * alpha + self.smoothed['pitch'] * (1 - alpha)
            self.smoothed['roll'] = roll_acc * alpha + self.smoothed['roll'] * (1 - alpha)
            self.smoothed['yaw'] = self._yaw

            # Keep some history for visualization/debug
            self.data_history.append((self.smoothed['pitch'], self.smoothed['roll'], self.smoothed['yaw']))

    def get_motion(self):
        """Return a motion dictionary with orientation and a movement vector.

        The movement vector is derived from pitch/roll mapped into the world X/Z plane
        and yaw rotation is provided so the character can face/move in the correct heading.
        """
        with self._lock:
            pitch = float(self.smoothed['pitch'])
            roll = float(self.smoothed['roll'])
            yaw = float(self.smoothed['yaw'])

            # Map device pitch/roll to forward/right motion. Tunable sensitivity.
            # Positive pitch => tilt forward => positive forward movement
            sensitivity = 0.05  # tuning knob
            forward = pitch * sensitivity
            right = roll * sensitivity

            # Rotate the local (forward,right) vector by yaw to get world x,z
            yaw_rad = math.radians(yaw)
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)

            vx = forward * sin_y + right * cos_y
            vz = forward * cos_y - right * sin_y

            # Use accelerometer Z to nudge vertical motion (small, optional)
            az = self.raw_data['accel'].get('Z', 0.0)
            # If device experiences a quick bump (az deviates from ~1g), allow small vertical response
            # Normalize expected gravity ~1.0 if data is g units; else keep 0
            vy = 0.0
            if az != 0.0:
                vy = (1.0 - az) * 0.2

            return {
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'pitch': pitch,
                'roll': roll,
                'yaw': yaw
            }

# ---------------------- 3D Character Game ----------------------
class Character3D:
    def __init__(self):
        # Character properties
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = 0.0
        self.scale = 1.0
        
        # Physics properties
        self.speed = 0.3
        self.jump_force = 0.5
        self.gravity = 0.02
        self.ground_level = 0.0
        self.is_jumping = False
        
        # Character colors
        self.colors = {
            'body': (0.2, 0.5, 0.8),
            'head': (0.8, 0.7, 0.6),
            'eyes': (0.1, 0.1, 0.1),
            'legs': (0.2, 0.3, 0.7)
        }

    def update(self, motion):
        """Update character position and physics"""
        # motion now contains vx, vy, vz and orientation: pitch, roll, yaw
        vx = motion.get('vx', 0.0)
        vy = motion.get('vy', 0.0)
        vz = motion.get('vz', 0.0)
        yaw = motion.get('yaw', None)

        # Apply forces -> set desired velocity and use inertia
        desired = np.array([vx, vy, vz], dtype=np.float32) * self.speed

        # Smoothly accelerate towards desired velocity (inertia)
        accel_factor = 0.15
        self.velocity += (desired - self.velocity) * accel_factor

        # Apply small drag/friction to horizontal motion
        drag = 0.02
        self.velocity[0] *= (1 - drag)
        self.velocity[2] *= (1 - drag)

        # Gravity always applies to Y
        if not self.is_jumping:
            # If character is on the ground, ensure small negative gravity keeps contact
            if self.position[1] <= self.ground_level + 0.001:
                self.velocity[1] = 0
            else:
                self.velocity[1] -= self.gravity

        # Integrate position
        self.position += self.velocity

        # Ground collision
        if self.position[1] < self.ground_level:
            self.position[1] = self.ground_level
            self.velocity[1] = 0
            self.is_jumping = False

        # Boundary limits (allow fairly large playing area)
        self.position[0] = np.clip(self.position[0], -40, 40)
        self.position[2] = np.clip(self.position[2], -40, 40)

        # Update rotation: prefer device yaw if available, otherwise infer from velocity
        if yaw is not None:
            self.rotation = yaw
        else:
            if abs(self.velocity[0]) > 0.01 or abs(self.velocity[2]) > 0.01:
                self.rotation = math.degrees(math.atan2(self.velocity[0], self.velocity[2]))

    def jump(self):
        """Make character jump"""
        if not self.is_jumping and self.position[1] <= self.ground_level:
            self.velocity[1] = self.jump_force
            self.is_jumping = True

    def draw(self):
        """Draw the 3D character"""
        glPushMatrix()
        
        # Position and rotate character
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glRotatef(self.rotation, 0, 1, 0)
        glScalef(self.scale, self.scale, self.scale)
        
        # Draw body
        glColor3f(*self.colors['body'])
        self._draw_box(0, 0.75, 0, 0.4, 0.75, 0.2)  # Torso
        
        # Draw head
        glColor3f(*self.colors['head'])
        self._draw_box(0, 1.75, 0, 0.3, 0.25, 0.3)  # Head
        
        # Draw eyes
        glColor3f(*self.colors['eyes'])
        self._draw_box(0.1, 1.8, -0.3, 0.08, 0.08, 0.01)  # Right eye
        self._draw_box(-0.1, 1.8, -0.3, 0.08, 0.08, 0.01)  # Left eye
        
        # Draw legs
        glColor3f(*self.colors['legs'])
        self._draw_box(-0.2, 0.3, 0, 0.1, 0.3, 0.1)  # Left leg
        self._draw_box(0.2, 0.3, 0, 0.1, 0.3, 0.1)  # Right leg
        
        # Draw arms
        self._draw_box(-0.5, 0.75, 0, 0.1, 0.3, 0.1)  # Left arm
        self._draw_box(0.5, 0.75, 0, 0.1, 0.3, 0.1)  # Right arm
        
        glPopMatrix()
        
        # Draw shadow
        self._draw_shadow()

    def _draw_box(self, x, y, z, width, height, depth):
        """Helper to draw a 3D box"""
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(width, height, depth)
        
        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-1, -1, 1)
        glVertex3f(1, -1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-1, -1, -1)
        glVertex3f(-1, 1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, -1, -1)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-1, 1, -1)
        glVertex3f(-1, 1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(1, 1, -1)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glVertex3f(1, -1, 1)
        glVertex3f(-1, -1, 1)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(1, -1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, 1, 1)
        glVertex3f(1, -1, 1)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-1, -1, -1)
        glVertex3f(-1, -1, 1)
        glVertex3f(-1, 1, 1)
        glVertex3f(-1, 1, -1)
        glEnd()
        
        glPopMatrix()

    def _draw_shadow(self):
        """Draw character shadow on ground"""
        glPushMatrix()
        glTranslatef(self.position[0], 0.01, self.position[2])
        glRotatef(self.rotation, 0, 1, 0)
        glScalef(self.scale, 0.01, self.scale)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)
        
        glColor4f(0.0, 0.0, 0.0, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(-0.4, 0, -0.2)
        glVertex3f(0.4, 0, -0.2)
        glVertex3f(0.4, 0, 0.2)
        glVertex3f(-0.4, 0, 0.2)
        glEnd()
        
        glEnable(GL_LIGHTING)
        glDisable(GL_BLEND)
        glPopMatrix()

class Game:
    def __init__(self):
        pygame.init()
        self.display = (1200, 800)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Character Game - MPU6050 Control")
        
        # OpenGL initialization
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 50.0)
        
        # Set up camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 10, 20,  # Camera position
                  0, 0, 0,    # Look at point
                  0, 1, 0)    # Up vector
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        self.character = Character3D()
        self.bt_controller = BluetoothController()
        
        # Try to connect to Bluetooth
        if not self.bt_controller.connect():
            print("⚠️  No Bluetooth connection - keyboard controls only")
        else:
            self.bt_controller.start()
            
        # Game state
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60

    def handle_input(self):
        """Handle keyboard and Bluetooth input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.character.jump()
        
        # Get motion from Bluetooth if connected, otherwise use keyboard
        if self.bt_controller.connected:
            motion = self.bt_controller.get_motion()
        else:
            keys = pygame.key.get_pressed()
            # Map keyboard arrows to vx/vz motion to match get_motion() shape
            forward = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 0.7
            right = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.7
            motion = {
                'vx': right,
                'vy': 0.0,
                'vz': -forward,
                'pitch': 0.0,
                'roll': 0.0,
                'yaw': None
            }
        
        return motion

    def draw_world(self):
        """Draw the 3D world"""
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Position camera
        gluLookAt(0, 10, 20,  # Camera position
                  self.character.position[0], self.character.position[1], self.character.position[2],  # Look at character
                  0, 1, 0)    # Up vector
        
        # Draw ground grid with slight tilt based on device pitch/roll for realism
        if self.bt_controller and self.bt_controller.connected:
            ori = self.bt_controller.get_motion()
            pitch = ori.get('pitch', 0.0)
            roll = ori.get('roll', 0.0)
        else:
            pitch = 0.0
            roll = 0.0

        tilt_scale = 0.6
        glPushMatrix()
        # Tilt around the character position so the field appears to tilt under the player
        cx, cz = self.character.position[0], self.character.position[2]
        glTranslatef(cx, 0, cz)
        glRotatef(roll * tilt_scale, 0, 0, 1)
        glRotatef(pitch * tilt_scale, 1, 0, 0)
        glTranslatef(-cx, 0, -cz)

        glBegin(GL_LINES)
        glColor3f(0.3, 0.3, 0.3)
        for i in range(-40, 41, 2):
            glVertex3f(i, 0, -40)
            glVertex3f(i, 0, 40)
            glVertex3f(-40, 0, i)
            glVertex3f(40, 0, i)
        glEnd()
        glPopMatrix()
        
        # Draw character
        self.character.draw()
        
        # Draw HUD
        self.draw_hud()
        
        pygame.display.flip()

    def draw_hud(self):
        """Draw HUD with game info"""
        # Create text surfaces
        font = pygame.font.Font(None, 36)
        
        if self.bt_controller.connected:
            status = "Connected to MPU6050"
            color = (0, 255, 0)
        else:
            status = "No Bluetooth - Using Keyboard"
            color = (255, 255, 0)
            
        text = font.render(status, True, color)
        
        # Convert surface to string buffer
        text_data = pygame.image.tostring(text, "RGBA", True)
        
        # Save current matrix
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for HUD
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Draw text
        glRasterPos2i(10, 30)
        glDrawPixels(text.get_width(), text.get_height(), 
                    GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore state
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def run(self):
        """Main game loop"""
        while self.running:
            motion = self.handle_input()
            self.character.update(motion)
            self.draw_world()
            self.clock.tick(self.fps)

        # Cleanup
        if self.bt_controller:
            self.bt_controller.stop()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    try:
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()