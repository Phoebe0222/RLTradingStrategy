hostname = 'localhost'
username = 'root'
password = 'Lyw12345'
database = 'sys'

import pymysql
db = pymysql.connect(host=hostname, user=username, passwd=password, db=database)

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")

# Fetch a single row using fetchone() method.
data = cursor.fetchone()
print ("Database version : %s " % data)

# disconnect from server
db.close()