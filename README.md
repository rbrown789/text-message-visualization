# A Python Tool for Parsing and Visualizing Facebook Message Conversations
This repository contains Python scripts which parse facebook message conversations, and then implement a high quality visualization of the conversation that contains:
  - word clouds and emoji clouds
  - line and radar plots summarizing message frequency by week, day of week, and hour of day
  - word and message summary statistics
  
## Contents
The contents of the repository are as follows:

  - **fbhtml_to_df.py**: A Python module for parsing an exported Facebook messenger HTML conversation. Running the script from the command line opens a dialog box to choose the file, and saves a CSV file of the same name with the parsed text, user, and date-timestamp.
  - **visualization_general.py**: A Python script for parsing a Facebook conversation and automatically generating a visualization.  Loads the **fbhtml_to_df.py** module. Running the script from the command line pens a dialog box to choose the HTML file, and saves a PNG file of the same name in the directory of the HTML file.
  - **fbmessage_example.html**: A real Facebook messenger conversation exported from Facebook with user identities masked. For use as a demo.
  - **Symbola.ttf**: An emoji-specific font file for rendering the emoji clouds.
  