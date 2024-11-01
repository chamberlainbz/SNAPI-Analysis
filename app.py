from shiny import App, ui, render, reactive, Inputs, Outputs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import numpy as np
import tempfile

# Constants for DK2 headset
DK2_RESOLUTION = (960, 1080)  # Per eye resolution in pixels
FOV_X = 90  # Horizontal field of view in degrees
FOV_Y = 100  # Vertical field of view in degrees

# Directory with participant data
default_directory = '/Users/f007kmv/Dartmouth College Dropbox/Benjamin Chamberlain Zivsak/Projects/SNAPI_BenVrGazeCore/SNAPI-Analysis/SNAPI-Analysis/rawDataNew_modified'

# Define the app UI with organized layout and extra spacing
app_ui = ui.page_fluid(
    ui.panel_title("Eye Center Visualization"),
    ui.input_select("participant_id", "Select Participant:", choices=[]),
    ui.input_slider("center_radius_deg", "Center Radius (°):", min=1, max=20, value=10, step=0.5),
    ui.input_file("file_upload", "Upload New File (optional):", multiple=False),
    
    # Individual Participant Plots
    ui.row(
        ui.column(12, ui.output_plot("fixation_plot", width="100%", height="500px")),
        ui.column(12, ui.output_plot("histogram_plot", width="100%", height="300px"))
    ),
    
    # Spacer between individual and aggregate plots
    ui.br(),
    ui.br(),

    # Aggregate Plots
    ui.row(
        ui.column(12, ui.output_plot("all_fixations_plot", width="100%", height="500px")),
        ui.column(12, ui.output_plot("all_histogram_plot", width="100%", height="300px"))
    )
)

# Define the server logic
def server(input: Inputs, output: Outputs, session):
    # Populate dropdown with participants from the default directory
    def load_participant_list():
        participant_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(default_directory)
            if f.endswith('.txt')
        ]
        ui.update_select("participant_id", choices=participant_files)
    
    load_participant_list()

    # Load and process data function
    def load_and_process_data(filepath):
        data = pd.read_csv(filepath, delimiter=',', header=None)
        data.columns = ['trial', 'date', 'core_time', 'exp_time', 'pitch', 'yaw', 'roll',
                        'right_x', 'right_y', 'left_x', 'left_y', 'right_conf', 'left_conf']
        
        # Calculate average eye position in normalized coordinates (0-1 range)
        data['eye_x'] = data[['right_x', 'left_x']].mean(axis=1)
        data['eye_y'] = data[['right_y', 'left_y']].mean(axis=1)
        
        # Scale to pixel coordinates
        data['pixel_x'] = data['eye_x'] * DK2_RESOLUTION[0]
        data['pixel_y'] = data['eye_y'] * DK2_RESOLUTION[1]
        
        return data

    # Reactive expression to load data
    @reactive.Calc
    def data():
        if input.file_upload() is not None:
            # If a file is uploaded, save it temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(input.file_upload()['data'])
            temp_file.close()
            filepath = temp_file.name
        else:
            # Otherwise, use the selected participant file
            filepath = os.path.join(default_directory, f"{input.participant_id()}.txt")
        
        return load_and_process_data(filepath)

    # Plot fixation positions
    @output
    @render.plot
    def fixation_plot():
        data_df = data()
        
        # Calculate center radius in pixels
        center_radius_pixels_x = DK2_RESOLUTION[0] * (input.center_radius_deg() / FOV_X)
        center_radius_pixels_y = DK2_RESOLUTION[1] * (input.center_radius_deg() / FOV_Y)
        
        # Plot fixation positions in DK2 pixel space
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data_df['pixel_x'], data_df['pixel_y'], s=5, color='purple', alpha=0.5, label='Fixations')
        ax.set_xlabel("Horizontal Position (pixels)")
        ax.set_ylabel("Vertical Position (pixels)")
        ax.invert_yaxis()
        ax.set_aspect('equal')

        # Add a central circle for DK2 headset center
        center_x, center_y = DK2_RESOLUTION[0] / 2, DK2_RESOLUTION[1] / 2
        center_circle = Circle((center_x, center_y), center_radius_pixels_x, color='red', alpha=0.3, label=f"Center ({input.center_radius_deg()}° radius)")
        ax.add_patch(center_circle)
        ax.set_xlim(0, DK2_RESOLUTION[0])
        ax.set_ylim(0, DK2_RESOLUTION[1])
        
        # Count fixations inside and outside the circle
        distances = np.sqrt((data_df['pixel_x'] - center_x) ** 2 + (data_df['pixel_y'] - center_y) ** 2)
        inside_count = np.sum(distances <= center_radius_pixels_x)
        outside_count = len(distances) - inside_count
        
        # Display counts in legend
        ax.legend(loc='upper right', title=f"Inside: {inside_count}, Outside: {outside_count}")
        ax.set_title(f"Eye Position Fixations on DK2 Headset Screen")

        return fig

    # Plot histogram for fixation proportions
    @output
    @render.plot
    def histogram_plot():
        data_df = data()
        center_radius_pixels_x = DK2_RESOLUTION[0] * (input.center_radius_deg() / FOV_X)
        center_x, center_y = DK2_RESOLUTION[0] / 2, DK2_RESOLUTION[1] / 2

        # Calculate proportion of fixations inside center circle
        distances = np.sqrt((data_df['pixel_x'] - center_x) ** 2 + (data_df['pixel_y'] - center_y) ** 2)
        inside_circle = distances <= center_radius_pixels_x
        inside_ratio = np.sum(inside_circle) / len(distances) if len(distances) > 0 else 0
        outside_ratio = 1 - inside_ratio

        # Plot histogram for proportions
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Inside Center", "Outside Center"], [inside_ratio, outside_ratio], color=['green', 'blue'])
        ax.set_ylabel("Proportion of Fixations")
        ax.set_title("Fixation Proportion Inside and Outside Center")

        # Add text labels for percentages
        for bar, percentage in zip(bars, [inside_ratio, outside_ratio]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{percentage:.1%}", ha='center', va='bottom'
            )
        
        return fig

    # Plot all participants' fixations
    @output
    @render.plot
    def all_fixations_plot():
        all_data = pd.DataFrame()
        participant_files = [f for f in os.listdir(default_directory) if f.endswith('.txt')]

        for file in participant_files:
            filepath = os.path.join(default_directory, file)
            participant_data = load_and_process_data(filepath)
            all_data = pd.concat([all_data, participant_data], ignore_index=True)
        
        # Calculate center radius in pixels based on input slider value
        center_radius_pixels_x = DK2_RESOLUTION[0] * (input.center_radius_deg() / FOV_X)
        
        # Plot fixation positions in DK2 pixel space
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(all_data['pixel_x'], all_data['pixel_y'], s=5, color='purple', alpha=0.5, label='Fixations')
        ax.set_xlabel("Horizontal Position (pixels)")
        ax.set_ylabel("Vertical Position (pixels)")
        ax.invert_yaxis()
        ax.set_aspect('equal')

        # Add a central circle for DK2 headset center
        center_x, center_y = DK2_RESOLUTION[0] / 2, DK2_RESOLUTION[1] / 2
        ax.add_patch(Circle((center_x, center_y), center_radius_pixels_x, color='red', alpha=0.3, label=f"Center ({input.center_radius_deg()}° radius)"))
        ax.set_xlim(0, DK2_RESOLUTION[0])
        ax.set_ylim(0, DK2_RESOLUTION[1])
        
        # Count fixations inside and outside the circle
        distances = np.sqrt((all_data['pixel_x'] - center_x) ** 2 + (all_data['pixel_y'] - center_y) ** 2)
        inside_count = np.sum(distances <= center_radius_pixels_x)
        outside_count = len(distances) - inside_count
        
        # Display counts in legend
        ax.legend(loc='upper right', title=f"Inside: {inside_count}, Outside: {outside_count}")
        ax.set_title("Aggregate Eye Position Fixations Across All Participants")

        return fig

    # Plot aggregate histogram for all participants
    @output
    @render.plot
    def all_histogram_plot():
        all_data = pd.DataFrame()
        participant_files = [f for f in os.listdir(default_directory) if f.endswith('.txt')]

        for file in participant_files:
            filepath = os.path.join(default_directory, file)
            participant_data = load_and_process_data(filepath)
            all_data = pd.concat([all_data, participant_data], ignore_index=True)
        
        # Calculate proportion of fixations inside center circle
        center_radius_pixels_x = DK2_RESOLUTION[0] * (input.center_radius_deg() / FOV_X)
        center_x, center_y = DK2_RESOLUTION[0] / 2, DK2_RESOLUTION[1] / 2
        distances = np.sqrt((all_data['pixel_x'] - center_x) ** 2 + (all_data['pixel_y'] - center_y) ** 2)
        inside_circle = distances <= center_radius_pixels_x
        inside_ratio = np.sum(inside_circle) / len(distances) if len(distances) > 0 else 0
        outside_ratio = 1 - inside_ratio

        # Plot aggregate histogram for proportions with different colors
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Inside Center", "Outside Center"], [inside_ratio, outside_ratio], color=['orange', 'purple'])
        ax.set_ylabel("Proportion of Fixations")
        ax.set_title("Aggregate Fixation Proportion Inside and Outside Center")

        # Add text labels for percentages
        for bar, percentage in zip(bars, [inside_ratio, outside_ratio]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{percentage:.1%}", ha='center', va='bottom'
            )

        return fig

app = App(app_ui, server)
