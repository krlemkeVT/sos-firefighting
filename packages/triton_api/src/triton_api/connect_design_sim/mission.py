from packages.triton_api.src.triton_api.connect_design_sim.setup import MissionSegment

# The complexity of the mission will be further developed but want to be sure the baseline works

def build_firefighting_mission():


    return [

        MissionSegment(
            name="taxi_out",
            duration_s=300
        ),


        MissionSegment(
            name="vertical_takeoff",
            duration_s=60,
            altitude_m=60,
            climb_rate_mps=7
        ),


        MissionSegment(
            name="transition",
            duration_s=20
        ),


        MissionSegment(
            name="cruise_climb",
            duration_s=120,
            altitude_m=500,
            climb_rate_mps=10
        ),


        MissionSegment(
            name="cruise",
            duration_s=1800,
            altitude_m=500,
            speed_mps=120
        ),


        MissionSegment(
            name="cruise_descent",
            duration_s=120,
            climb_rate_mps=-5
        ),


        MissionSegment(
            name="retransition",
            duration_s=20
        ),


        MissionSegment(
            name="hover_loiter",
            duration_s=600,
            speed_mps=46.6
        ),


        MissionSegment(
            name="landing",
            duration_s=60,
            altitude_m=60,
            climb_rate_mps=-7
        ),


        MissionSegment(
            name="scoop",
            duration_s=300
        )

    ]