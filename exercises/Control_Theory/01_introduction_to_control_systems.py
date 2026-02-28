"""
Exercises for Lesson 01: Introduction to Control Systems
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: System Identification
    For each system, identify reference input, plant, sensor, disturbance, and controller.
    """
    systems = [
        {
            "name": "Air conditioning system maintaining room temperature at 22°C",
            "reference": "Desired temperature setpoint = 22°C",
            "plant": "Room (thermal mass, insulation, air volume)",
            "sensor": "Temperature sensor (thermocouple or thermistor)",
            "disturbance": "Outside temperature changes, people entering/leaving, "
                           "doors opening, solar radiation",
            "controller": "Thermostat / HVAC control unit that decides when to "
                          "heat or cool based on temperature error",
        },
        {
            "name": "Self-driving car maintaining a lane on a highway",
            "reference": "Desired lane center position (lateral offset = 0)",
            "plant": "Vehicle dynamics (steering, tires, chassis)",
            "sensor": "Camera, LiDAR, lane-detection algorithms providing "
                      "lateral position and heading error",
            "disturbance": "Crosswind, road curvature, uneven road surface, "
                           "other vehicles",
            "controller": "Autonomous driving ECU (electronic control unit) "
                          "computing steering angle from lane position error",
        },
        {
            "name": "Voltage regulator keeping output at 5V",
            "reference": "Desired output voltage = 5V",
            "plant": "Power supply circuit (transformer, rectifier, filter)",
            "sensor": "Voltage divider / feedback network measuring output voltage",
            "disturbance": "Input voltage fluctuations (line regulation), "
                           "load current changes (load regulation)",
            "controller": "Error amplifier and PWM modulator adjusting duty cycle "
                          "to regulate output voltage",
        },
    ]

    for i, sys in enumerate(systems, 1):
        print(f"\nSystem {i}: {sys['name']}")
        print(f"  Reference:   {sys['reference']}")
        print(f"  Plant:       {sys['plant']}")
        print(f"  Sensor:      {sys['sensor']}")
        print(f"  Disturbance: {sys['disturbance']}")
        print(f"  Controller:  {sys['controller']}")


def exercise_2():
    """
    Exercise 2: Open-Loop vs. Closed-Loop
    """
    print("\n1. Washing machine with fixed 30-minute cycle:")
    print("   This is OPEN-LOOP control.")
    print("   It runs for a predetermined time regardless of how clean the clothes are.")
    print("   To make it CLOSED-LOOP: add a turbidity sensor to measure water clarity.")
    print("   The machine would continue washing until the water is sufficiently clear,")
    print("   then stop automatically.")

    print("\n2. Why cruise control cannot be purely open-loop on hilly terrain:")
    print("   On hilly terrain, the load on the engine varies (gravity helps going")
    print("   downhill, opposes going uphill). An open-loop controller applies a fixed")
    print("   throttle position calculated for flat road. Going uphill, the car slows")
    print("   down; going downhill, it speeds up. Without measuring actual speed and")
    print("   adjusting throttle (feedback), the car cannot maintain constant speed.")

    print("\n3. Example where open-loop is preferred over closed-loop:")
    print("   A traffic light system with fixed timing schedules.")
    print("   Reasons: simple, reliable, no sensor needed, no instability risk,")
    print("   and the timing can be optimized offline based on traffic patterns.")
    print("   (Though adaptive traffic lights using sensors do exist for busy cities.)")


def exercise_3():
    """
    Exercise 3: Feedback Effects
    Plant gain Gp varies between 8 and 12 (nominal = 10).
    """
    Gp_values = np.array([8.0, 10.0, 12.0])

    # Part 1: Open-loop with Gc = 0.1
    print("Part 1: Open-loop control (Gc = 0.1)")
    Gc_open = 0.1
    gains_open = Gc_open * Gp_values
    print(f"  Gp = {Gp_values}")
    print(f"  Overall gain = Gc * Gp = {gains_open}")
    print(f"  Output range: {gains_open[0]:.2f} to {gains_open[2]:.2f}")
    print(f"  Variation: +/-{(gains_open[2] - gains_open[0]) / gains_open[1] * 50:.1f}% "
          f"around nominal")

    # Part 2: Closed-loop with Gc = 100, H = 1
    print("\nPart 2: Closed-loop control (Gc = 100, H = 1)")
    Gc_closed = 100.0
    H = 1.0
    gains_closed = (Gc_closed * Gp_values) / (1 + Gc_closed * Gp_values * H)
    for Gp, gain_cl in zip(Gp_values, gains_closed):
        loop_gain = Gc_closed * Gp
        print(f"  Gp = {Gp:5.1f}: CL gain = {Gc_closed}*{Gp}/(1+{Gc_closed}*{Gp}) "
              f"= {loop_gain}/{1+loop_gain:.0f} = {gain_cl:.6f}")

    variation_pct = (gains_closed[2] - gains_closed[0]) / gains_closed[1] * 100
    print(f"  Output range: {gains_closed[0]:.6f} to {gains_closed[2]:.6f}")
    print(f"  Variation: {variation_pct:.3f}% around nominal")
    print(f"  Compare: open-loop variation = 40%, closed-loop variation = {variation_pct:.3f}%")

    # Part 3: Cost of high loop gain
    print("\nPart 3: Cost of high loop gain for reducing sensitivity")
    print("  - Requires a high-gain amplifier (Gc = 100), which may be expensive")
    print("  - High gain amplifies sensor noise: noise is multiplied by Gc")
    print("  - Risk of instability: high loop gain reduces phase margin")
    print("  - Increased control effort: actuator must handle larger signals")
    print("  - Bandwidth increases, making the system more susceptible to")
    print("    unmodeled high-frequency dynamics")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: System Identification ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Open-Loop vs. Closed-Loop ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Feedback Effects ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
