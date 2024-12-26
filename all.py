from __future__ import division  # Import division from __future__
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




miami = pd.read_csv("")
pd.set_option('display.max_columns', None)
# print(miami.dtypes)



miami['DOS_1'].dropna()
miami['Winning Bid'] = miami['Winning Bid'].str.replace('$', '')
miami['Winning Bid'] = miami['Winning Bid'].str.replace(',', '').astype(float)

miami_clean = miami[(miami['PRICE_1'] >= 1000) & (miami['PRICE_1'] >= 0)]

# mask = miami_clean['PRICE_1'] != miami_clean['Winning Bid']

# miami_clean2 = miami_clean[mask]

miami_clean.to_csv("")

cleanm = pd.read_csv("")
cleanm['Sale Date'] = pd.to_datetime(cleanm['Sale Date'])
cleanm['DOS_1'] = pd.to_datetime(cleanm['DOS_1'], format='%Y%m%d')
# print(cleanm.head())
# print(cleanm.dtypes)
cleanm['Flip_Days'] = (cleanm['DOS_1'] - cleanm['Sale Date']).dt.days
cleanm['Profit'] = (cleanm['PRICE_1'] - cleanm['Winning Bid'])
cleanm['Profit'].dropna()
cleanm['ROI'] = (cleanm['Profit']/cleanm['Winning Bid'])*100

values_to_drop = ['OFFICE BUILDING - ONE STORY : OFFICE BUILDING', 'VACANT RESIDENTIAL : EXTRA FEA OTHER THAN PARKING', 'STORE : CONDOMINIUM - COMMERCIAL', 'VACANT RESIDENTIAL : VACANT LAND',
                  'MULTIFAMILY 2-9 UNITS : 2 LIVING UNITS', 'VACANT LAND - INDUSTRIAL : EXTRA FEA OTHER THAN PARKING', 'RESTAURANT OR CAFETERIA : RETAIL OUTLET', 'HOTEL OR MOTEL : CONDOMINIUM', 
                  'WAREHOUSE TERMINAL OR STG : WAREHOUSE OR STORAGE', 'MULTIFAMILY 2-9 UNITS : MULTIFAMILY 3 OR MORE UNITS', 'WAREHOUSE TERMINAL OR STG : CONDOMINIUM - COMMERCIAL',
                  'ACREAGE NOT CLASSIFIED AG : VACANT LAND', 'MULTIFAMILY 10 UNITS PLUS : MULTIFAMILY 3 OR MORE UNITS', 'OFFICE BUILDING - MULTISTORY : CONDOMINIUM - COMMERCIAL', 
                  'LIGHT MANUFACTURING : CONDOMINIUM - COMMERCIAL', 'VACANT LAND - COMMERCIAL : VACANT LAND', 'IMPR AGRI - NOT HOMESITES : 2 LIVING UNITS']
# Drop rows where 'Column1' has any of the specified values
cleanma = cleanm[~cleanm['DOR_DESC'].isin(values_to_drop)]
# print(cleanma.head(20))

# Group by 'TRUE_SITE_CITY'
grouped = cleanma.groupby('TRUE_SITE_CITY')
# Calculate percentage of 'Plaintiff' and other bidders
total_bidders = grouped['Bidder'].count()
plaintiff_count = grouped.apply(lambda x: (x['Bidder'] == 'Plaintiff').sum())
other_bidders_count = total_bidders - plaintiff_count
percentage_plaintiff = (plaintiff_count / total_bidders * 100)
percentage_other_bidders = 100 - percentage_plaintiff

# Format percentage values
percentage_plaintiff_formatted = percentage_plaintiff.map('{:.2f}%'.format)
percentage_other_bidders_formatted = percentage_other_bidders.map('{:.2f}%'.format)

# Create DataFrames for each result with city names as index
df_other_bidders = pd.DataFrame({'Other Bidders': percentage_other_bidders_formatted}, index=percentage_other_bidders_formatted.index)
df_plaintiff = pd.DataFrame({'Plaintiff': percentage_plaintiff_formatted}, index=percentage_plaintiff_formatted.index)

# Concatenate the DataFrames along the columns axis
combined_df = pd.concat([df_other_bidders, df_plaintiff], axis=1)

# Print the combined DataFrame
print(combined_df)

cleanmab = cleanma[cleanma['Bidder'] != 'Plaintiff']
cleanmab.to_csv("")

import pandas as pd
pd.set_option('display.max_columns', None)
# Missing: Add to plot for each year,  Correlation by City by Year, Scattered Plot to show relationship in heatmap.
pd.set_option('display.max_rows', None)
df = pd.read_csv("")
df['Sale Date'] = pd.to_datetime(df['Sale Date'])
df['DOS_1'] = pd.to_datetime(df['DOS_1'])

# Create New Calculations
df['Purchase Date'] = df['Sale Date'].dt.year
df['Purchase Sq Ft'] = df['Winning Bid']/df['BUILDING_ACTUAL_AREA']
df['Sale Sq Ft'] = df['PRICE_1']/df['BUILDING_ACTUAL_AREA']

bedrooms = range(int(df['BEDROOM_COUNT'].min()), int(df['BEDROOM_COUNT'].max()) + 1)
bathrooms = range(int(df['BATHROOM_COUNT'].min()), int(df['BATHROOM_COUNT'].max()) + 1)
half_bathrooms = range(int(df['HALF_BATHROOM_COUNT'].min()), int(df['HALF_BATHROOM_COUNT'].max()) + 1)


def calculate_profit_stats(df):
    # Initialize an empty DataFrame to store results
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            for bedroom in bedrooms:
                for bathroom in bathrooms:
                    for half_bathroom in half_bathrooms:
                        # Filter the DataFrame based on the current combination of criteria
                        filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city) &
                                         (df['BEDROOM_COUNT'] == bedroom) & (df['BATHROOM_COUNT'] == bathroom) &
                                         (df['HALF_BATHROOM_COUNT'] == half_bathroom)]
                        
                        # Calculate statistics for 'Profit' column
                        min_profit = filtered_df['Profit'].min()
                        max_profit = filtered_df['Profit'].max()
                        avg_profit = filtered_df['Profit'].mean()
                        q75_flipdays = filtered_df['Profit'].quantile(0.75)
                        cto_profit = filtered_df['Profit'].count()
                        
                        # Create a dictionary representing the current row
                        row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'BEDROOM_COUNT': bedroom,
                                    'BATHROOM_COUNT': bathroom, 'HALF_BATHROOM_COUNT': half_bathroom,
                                    'Min_Profit': min_profit, 'Max_Profit': max_profit,
                                    'Average_Profit': avg_profit, '75th Quartile': q75_flipdays, 'Total Count': cto_profit}
                        
                        # Append the dictionary to the list
                        results_data.append(row_dict)

                        # Print the progress
                        # print(f"Processed Profit: Year={year}, City={city}, Bedrooms={bedroom}, Bathrooms={bathroom}, Half Bathrooms={half_bathroom}")

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv("", index=False)


def calculate_psqft_stats(df):
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            for bedroom in range(int(df['BEDROOM_COUNT'].min()), int(df['BEDROOM_COUNT'].max()) + 1):
                for bathroom in range(int(df['BATHROOM_COUNT'].min()), int(df['BATHROOM_COUNT'].max()) + 1):
                    for half_bathroom in range(int(df['HALF_BATHROOM_COUNT'].min()), int(df['HALF_BATHROOM_COUNT'].max()) + 1):
                        # Filter the DataFrame based on the current combination of criteria
                        filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city) &
                                         (df['BEDROOM_COUNT'] == bedroom) & (df['BATHROOM_COUNT'] == bathroom) &
                                         (df['HALF_BATHROOM_COUNT'] == half_bathroom)]
                        
                        # Calculate statistics for 'Profit' column
                        min_psqft = filtered_df['Purchase Sq Ft'].min()
                        max_psqft = filtered_df['Purchase Sq Ft'].max()
                        avg_psqft = filtered_df['Purchase Sq Ft'].mean()
                        q75_flipdays = filtered_df['Purchase Sq Ft'].quantile(0.75)
                        cto_psqft = filtered_df['Purchase Sq Ft'].count()
                        
                        # Create a dictionary representing the current row
                        row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'BEDROOM_COUNT': bedroom,
                                    'BATHROOM_COUNT': bathroom, 'HALF_BATHROOM_COUNT': half_bathroom,
                                    'Min_Purchase_SqFt': min_psqft, 'Max_Purchase_SqFt': max_psqft,
                                    'Average_Purchase_SqFt': avg_psqft, '75th Quartile': q75_flipdays, 'Total Count': cto_psqft}
                        
                        # Append the dictionary to the list
                        results_data.append(row_dict)

                        # Print the progress
                        # print(f"Processed Purchase Sq Ft: Year={year}, City={city}, Bedrooms={bedroom}, Bathrooms={bathroom}, Half Bathrooms={half_bathroom}")

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv('', index=False)

def calculate_ssqft_stats(df):
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            for bedroom in range(int(df['BEDROOM_COUNT'].min()), int(df['BEDROOM_COUNT'].max()) + 1):
                for bathroom in range(int(df['BATHROOM_COUNT'].min()), int(df['BATHROOM_COUNT'].max()) + 1):
                    for half_bathroom in range(int(df['HALF_BATHROOM_COUNT'].min()), int(df['HALF_BATHROOM_COUNT'].max()) + 1):
                        # Filter the DataFrame based on the current combination of criteria
                        filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city) &
                                         (df['BEDROOM_COUNT'] == bedroom) & (df['BATHROOM_COUNT'] == bathroom) &
                                         (df['HALF_BATHROOM_COUNT'] == half_bathroom)]
                        
                        # Calculate statistics for 'Profit' column
                        min_ssqft = filtered_df['Sale Sq Ft'].min()
                        max_ssqft = filtered_df['Sale Sq Ft'].max()
                        avg_ssqft = filtered_df['Sale Sq Ft'].mean()
                        q75_flipdays = filtered_df['Sale Sq Ft'].quantile(0.75)
                        cto_ssqft = filtered_df['Sale Sq Ft'].count()
                        
                        # Create a dictionary representing the current row
                        row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'BEDROOM_COUNT': bedroom,
                                    'BATHROOM_COUNT': bathroom, 'HALF_BATHROOM_COUNT': half_bathroom,
                                    'Min_Sale_SqFt': min_ssqft, 'Max_Sale_SqFt': max_ssqft,
                                    'Average_Sale_SqFt': avg_ssqft, '75th Quartile': q75_flipdays, 'Total Count': cto_ssqft}
                        
                        # Append the dictionary to the list
                        results_data.append(row_dict)

                        # Print the progress
                        # print(f"Processed Sale Sq Ft: Year={year}, City={city}, Bedrooms={bedroom}, Bathrooms={bathroom}, Half Bathrooms={half_bathroom}")

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv('', index=False)

def calculate_flipdays_stats(df):
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            for bedroom in range(int(df['BEDROOM_COUNT'].min()), int(df['BEDROOM_COUNT'].max()) + 1):
                for bathroom in range(int(df['BATHROOM_COUNT'].min()), int(df['BATHROOM_COUNT'].max()) + 1):
                    for half_bathroom in range(int(df['HALF_BATHROOM_COUNT'].min()), int(df['HALF_BATHROOM_COUNT'].max()) + 1):
                        # Filter the DataFrame based on the current combination of criteria
                        filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city) &
                                         (df['BEDROOM_COUNT'] == bedroom) & (df['BATHROOM_COUNT'] == bathroom) &
                                         (df['HALF_BATHROOM_COUNT'] == half_bathroom)]
                        
                        # Calculate statistics for 'Profit' column
                        min_flipdays = filtered_df['Flip_Days'].min()
                        max_flipdays = filtered_df['Flip_Days'].max()
                        avg_flipdays = filtered_df['Flip_Days'].mean()
                        med_flipdays = filtered_df['Flip_Days'].median()
                        q75_flipdays = filtered_df['Flip_Days'].quantile(0.75)
                        cto_flipdays = filtered_df['Flip_Days'].count()
                        
                        # Create a dictionary representing the current row
                        row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'BEDROOM_COUNT': bedroom,
                                    'BATHROOM_COUNT': bathroom, 'HALF_BATHROOM_COUNT': half_bathroom,
                                    'Min_FlipDays': min_flipdays, 'Max_FlipDays': max_flipdays,
                                    'Average_FlipDays': avg_flipdays, 'Median Flipdays': med_flipdays, '75th Quartile': q75_flipdays, 'Total Count': cto_flipdays}
                        
                        # Append the dictionary to the list
                        results_data.append(row_dict)

                        # Print the progress
                        # print(f"Processed Flip Days: Year={year}, City={city}, Bedrooms={bedroom}, Bathrooms={bathroom}, Half Bathrooms={half_bathroom}")

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv('', index=False)


calculate_profit_stats(df)
calculate_psqft_stats(df)
calculate_ssqft_stats(df)
calculate_flipdays_stats(df)


def calculate_profit_stats_gen(df):
    # Initialize an empty DataFrame to store results
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            # Filter the DataFrame based on the current combination of criteria
            filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city)]
                        
            # Calculate statistics for 'Profit' column
            min_profit = filtered_df['Profit'].min()
            max_profit = filtered_df['Profit'].max()
            avg_profit = filtered_df['Profit'].mean()
            q75_flipdays = filtered_df['Profit'].quantile(0.75)
            cto_profit = filtered_df['Profit'].count()
                        
            # Create a dictionary representing the current row
            row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'Min_Profit': min_profit, 'Max_Profit': max_profit,
                        'Average_Profit': avg_profit, '75th Quartile': q75_flipdays, 'Total Count': cto_profit}
                        
            # Append the dictionary to the list
            results_data.append(row_dict)
            
    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv("", index=False)

    # Plotting
    sns.catplot(x='TRUE_SITE_CITY', y='Average_Profit', hue='Year', data=results_df, kind='bar', height=6, aspect=2)
    plt.title('Average Profit by City & Year')
    plt.ylabel('Average Profit')
    plt.xlabel('City')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot
    plt.savefig('')
    plt.show()

def calculate_psqft_stats_gen(df):
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            # Filter the DataFrame based on the current combination of criteria
            filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city)]
                        
            # Calculate statistics for 'Profit' column
            min_psqft = filtered_df['Purchase Sq Ft'].min()
            max_psqft = filtered_df['Purchase Sq Ft'].max()
            avg_psqft = filtered_df['Purchase Sq Ft'].mean()
            q75_flipdays = filtered_df['Purchase Sq Ft'].quantile(0.75)
            cto_psqft = filtered_df['Purchase Sq Ft'].count()
                        
            # Create a dictionary representing the current row
            row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'Min_Purchase_SqFt': min_psqft, 'Max_Purchase_SqFt': max_psqft,
                        'Average_Purchase_SqFt': avg_psqft, '75th Quartile': q75_flipdays, 'Total Count': cto_psqft}
                        
            # Append the dictionary to the list
            results_data.append(row_dict)


    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv('', index=False)

def calculate_ssqft_stats_gen(df):
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            # Filter the DataFrame based on the current combination of criteria
            filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city)]
                        
            # Calculate statistics for 'Profit' column
            min_ssqft = filtered_df['Sale Sq Ft'].min()
            max_ssqft = filtered_df['Sale Sq Ft'].max()
            avg_ssqft = filtered_df['Sale Sq Ft'].mean()
            q75_flipdays = filtered_df['Sale Sq Ft'].quantile(0.75)
            cto_ssqft = filtered_df['Sale Sq Ft'].count()
                        
            # Create a dictionary representing the current row
            row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'Min_Sale_SqFt': min_ssqft, 'Max_Sale_SqFt': max_ssqft,
                        'Average_Sale_SqFt': avg_ssqft, '75th Quartile': q75_flipdays, 'Total Count': cto_ssqft}
                        
            # Append the dictionary to the list
            results_data.append(row_dict)

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv('', index=False)

def calculate_flipdays_stats_gen(df):
    # Initialize an empty list to store dictionaries
    results_data = []

    # Iterate through unique combinations of 'Year' and 'TRUE_SITE_CITY'
    for year in df['Purchase Date'].unique():
        for city in df['TRUE_SITE_CITY'].unique():
            # Filter the DataFrame based on the current combination of criteria
            filtered_df = df[(df['Purchase Date'] == year) & (df['TRUE_SITE_CITY'] == city)]
                        
            # Calculate statistics for 'Profit' column
            min_flipdays = filtered_df['Flip_Days'].min()
            max_flipdays = filtered_df['Flip_Days'].max()
            avg_flipdays = filtered_df['Flip_Days'].mean()
            med_flipdays = filtered_df['Flip_Days'].median()
            q75_flipdays = filtered_df['Flip_Days'].quantile(0.75)
            cto_flipdays = filtered_df['Flip_Days'].count()
                        
            # Create a dictionary representing the current row
            row_dict = {'Year': year, 'TRUE_SITE_CITY': city, 'Min_FlipDays': min_flipdays, 'Max_FlipDays': max_flipdays,
                        'Average_FlipDays': avg_flipdays, 'Median Flipdays': med_flipdays, '75th Quartile': q75_flipdays, 'Total Count': cto_flipdays}
                        
            # Append the dictionary to the list
            results_data.append(row_dict)

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_data)

    # Drop rows where 'Total_Count' is 0
    results_df = results_df[results_df['Total Count'] != 0]

    # Save the results to a CSV file
    results_df.to_csv('', index=False)

calculate_profit_stats_gen(df)
calculate_psqft_stats_gen(df)
calculate_ssqft_stats_gen(df)
calculate_flipdays_stats_gen(df)

df.to_csv("")


