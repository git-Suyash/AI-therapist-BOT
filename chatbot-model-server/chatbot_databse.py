import sqlite3

def create_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('chatbot_database.db')
    cursor = conn.cursor()

    # Create products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        price REAL NOT NULL
    )
    ''')

    # Create interactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Insert some example product data
    cursor.execute('''
    INSERT INTO products (name, description, price) VALUES
    ('xyz', 'Delicious cheese pizza', 9.99),
    ('abc', 'Juicy beef burger with lettuce and tomato', 7.99),
    ('Sushi', 'Fresh sushi rolls with wasabi and soy sauce', 12.99)
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database created and example data inserted.")
