import sqlite3
import short_url

def create_table(cursorObj):
    cursorObj.execute("""
            CREATE TABLE url_lookup(
                url_id INTEGER PRIMARY KEY,
                url VARCHAR(255) NOT NULL,
                short_code VARCHAR(255) UNIQUE NOT NULL
                );
           """)

def determine_next_index(cursorObj):
   cursorObj.execute("""
               SELECT COUNT(url_id) FROM url_lookup
               """)
   return cursorObj.fetchone()[0] + 1

def check_if_table_exists(cursorObj, table):
    ret_val = list_tables(cursorObj)
    if ret_val is not None:
        return True
    else:
        return False


def list_tables(cursorObj):
    #cursorObj = con.cursor()
    cursorObj.execute('SELECT name from sqlite_master where type= "table"')
    return cursorObj.fetchone()


def encode_url(conn,
               long_url,
               database = 'assets/url.db',
               table_name = 'url_lookup'):
    cursorObj = conn.cursor()
    if not check_if_table_exists(cursorObj, table_name):
        create_table(cursorObj)
    select_t = (long_url,)
    cursorObj.execute("""
                      SELECT url_id FROM url_lookup
                      WHERE url = ?
                      """, select_t)
    ret_val = cursorObj.fetchone()
    #print(f"ret_val : {ret_val}")
    if ret_val is None:
        next_index = determine_next_index(cursorObj)
        short_code = short_url.encode_url(next_index)
        insert_t = (next_index, long_url, short_code,)
        cursorObj.execute("""
                    INSERT INTO url_lookup(url_id, url, short_code)
                    VALUES(?, ?, ?);
                  """, insert_t)
    else:
        short_code = short_url.encode_url(ret_val[0])
    #print(f"short_code : {short_code}")
    cursorObj.close()
    return short_code

def open_connection(database='assets/url.db'):
    return sqlite3.connect(database)

def decode_url(conn,
              short_code,
              database = 'assets/url.db',
               table_name = 'url_lookup'):
    cursorObj = conn.cursor()
    #print(f"short_code : {short_code}")
    db_idx = short_url.decode_url(short_code)
    #print(f"db_idx : {db_idx}")
    select_t = (db_idx,)
    cursorObj.execute("""
                      SELECT url FROM url_lookup
                      WHERE url_id = ?
                      """, select_t)
    long_url = cursorObj.fetchone()[0]
    #print(f"long_url : {long_url}")
    cursorObj.close()
    return long_url

if __name__ == "__main__":
	long_url = "?yscale_rb=linear&normalise=simple&threshold_cumulative=0&threshold_daily=0&rollingMean=0&country_names=%5B32%2C+55%2C+59%2C+76%2C+82%2C+137%2C+153%2C+155%5D&cv_variables=1&date_picker=%5B%272020-01-22%27%2C+%272020-06-02%27%5D"
	database = 'assets/url.db'

	print(f"long_url : {long_url}")
	conn = open_connection(database='assets/url.db')
	short_code = encode_url(conn, long_url)
	print(f"short_code : {short_code}")
	ret_url = decode_url(conn, short_code)
	print(f"long_url : {ret_url}")
	conn.close()
