from pyspark import SparkContext

def preprocess_line(line):
    parts = line.split(',')
    date = parts[0].strip('"') # Assuming the date is at index 0
    customer_id = parts[1].strip('"').lstrip('0')  # Remove leading zeros and quotes
    product_id_str = parts[5].strip('"').lstrip('0')  # Remove quotes and leading zeros
    product_id = int(product_id_str) if product_id_str.isdigit() else 0  # Convert to int, default to 0 if not a digit
    date_customer_id = f"{date}-{customer_id}"  # Combine date and customer ID
    return (date_customer_id, product_id)


def write_to_csv(output_data, output_csv_path):
    with open(output_csv_path, 'w') as file:
        # Write the header
        file.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
        # Write each line of data
        for line in output_data:
            # Ensure PRODUCT_ID is written as an integer
            file.write(f"{line[0]},{line[1]}\n")

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    # Load and preprocess the data
    rdd = sc.textFile('/Users/kailongwang/dsci553/a2/ta_feng_all_months_merged.csv')
    header = rdd.first()
    rdd = rdd.filter(lambda line: line != header).map(preprocess_line)

    # Aggregate 'PRODUCT_ID' into a list for each 'DATE-CUSTOMER_ID'
    rdd_baskets = rdd.groupByKey().mapValues(list)

    # Flatten the list to ensure each product ID is on a separate line
    rdd_output = rdd_baskets.flatMap(lambda kv: [(kv[0], prod) for prod in kv[1]])

    # Collect the output
    output_data = rdd_output.collect()

    # Specify the path for the output CSV file
    output_csv_path = '/Users/kailongwang/dsci553/a2/preprocessed_ta_feng.csv'

    # Write the preprocessed data to a CSV file
    write_to_csv(output_data, output_csv_path)

    print(f'Data written to {output_csv_path}')

    # Stop the Spark context
    sc.stop()

