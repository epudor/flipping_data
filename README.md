Introduction

This script analyzes real estate flipping data in a CSV file. It calculates various metrics including profit, purchase price per square foot, sale price per square foot, and average flip days. The results can be used to understand market trends and identify profitable investment opportunities.

Requirements

Python 3.x
pandas
seaborn
matplotlib
Instructions

Replace the placeholders (denoted by "") in the script with the actual file paths for your data and output files.
Run the script: python real_estate_analysis.py
Functionalities:

Detailed Reports:
calculate_profit_stats: Generates a report with profit statistics (min, max, average, 75th percentile) by city and year, for various property configurations (bedrooms, bathrooms, half bathrooms).
calculate_psqft_stats: Generates a report with purchase price per square foot statistics by city and year, for various property configurations.
calculate_ssqft_stats: Generates a report with sale price per square foot statistics by city and year, for various property configurations.
calculate_flipdays_stats: Generates a report with average, median, and other flip day statistics by city and year, for various property configurations.
Visualization:
calculate_profit_stats_gen: Generates a bar chart showing average profit by city and year.
Notes:

The script handles missing values in the "DOS_1" column.
It filters out rows with specific property types listed in the values_to_drop variable.
You can adjust the script to customize the calculations and visualizations.
Disclaimer

This script is provided for informational purposes only and should not be considered financial advice. Always conduct your own research before making any investment decisions. 1 
