import sqlite3 
import pickle



# Sample new detection
new_detection = [
    ["sweatshirt|50%", "upperbody", "", "black"],
    ["sleeve|50%", "garment parts", "plain\nsymmetrical\nset-in sleeve", "darkslategray"],
    ["sleeve|50%", "garment parts", "wrist-length\nplain\nsymmetrical\nset-in sleeve", "black"]
]

# Connect to SQLite database
_conn = sqlite3.connect('db.sqlite3')
_cur = _conn.cursor()

# Retrieve data from the database
_cur.execute('SELECT * FROM product_data')
db_results = _cur.fetchall()
if not db_results:
    print("No data available at the moment! Try again later")
else:
    # Calculate rating and select highest rated images
    selected_results = calculate_rating(new_detection, db_results)
    # Print selected results
    print("Selected Results:")
    for result in selected_results:
        print(result)
    print(len(result))

# Close connection
_conn.close()

