import psycopg2


def get_connection():

    # Establish a connection to the PostgreSQL database
    connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5432,  # The port you exposed in docker-compose.yml
        database="mydb",
    )

    return connect
