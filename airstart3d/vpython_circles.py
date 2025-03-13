from vpython import scene, sphere, vector, rate, color, cylinder, text
import math

# Simulation parameters
total_time = 300  # total simulation time in seconds (5 minutes)
dt = 0.05         # time step for the simulation
scene.width = 1920-50
scene.height = 1080-100

# Define parameters for each pilot
n_pilots = 3
centers = [vector(-10, -10, 0), vector(10, -10, 0), vector(0, 10, 0)]
radii = [10, 15, 20]
angular_velocities = [0.5, 0.3, 0.2]  # radians per second for each pilot
altitudes = [50, 60, 70]  # constant altitude (z-coordinate) for each pilot

# Create spheres for each pilot
pilots = []
colors_list = [color.red, color.green, color.blue]
for i in range(n_pilots):
    # Starting position: at angle 0 on the circle for each pilot
    start_x = centers[i].x + radii[i]
    start_y = centers[i].y
    start_z = altitudes[i]
    sp = sphere(pos=vector(start_x, start_y, start_z),
                radius=1,
                make_trail=True, retain=100, trail_radius=0.1,
                color=colors_list[i])
    pilots.append(sp)

# Add axes:
L = 50
R = L/100
d = L-2
xaxis = cylinder(pos=vector(0,0,0), axis=vector(d,0,0), radius=R, color=color.yellow)
yaxis = cylinder(pos=vector(0,0,0), axis=vector(0,d,0), radius=R, color=color.yellow)
zaxis = cylinder(pos=vector(0,0,0), axis=vector(0,0,d), radius=R, color=color.yellow)
k = 1.02
h = 0.05*L
text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h, align='center', billboard=True, emissive=True)
text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h, align='center', billboard=True, emissive=True)
text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h, align='center', billboard=True, emissive=True)

# Animation loop
t = 0
while t < total_time:
    rate(1/dt)  # controls the simulation speed
    for i in range(n_pilots):
        # Calculate the new angle for this pilot
        theta = angular_velocities[i] * t
        # Update x and y to follow a circular path
        new_x = centers[i].x + radii[i] * math.cos(theta)
        new_y = centers[i].y + radii[i] * math.sin(theta)
        # Update sphere position (altitude remains constant)
        pilots[i].pos = vector(new_x, new_y, altitudes[i])

        # Change the color based on the angle:
        # r, g, b values vary with cosine and sine functions to produce a smooth transition.
        r_val = (math.cos(theta) + 1) / 2    # Normalize cosine to [0,1]
        g_val = (math.sin(theta) + 1) / 2      # Normalize sine to [0,1]
        b_val = (math.cos(theta + math.pi/2) + 1) / 2  # Phase-shifted cosine for variation
        
        pilots[i].color = vector(r_val, g_val, b_val)

    t += dt
