package upa.model;

import oracle.jdbc.OraclePreparedStatement;
import oracle.jdbc.pool.OracleDataSource;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Arrays;

/**
 * Class that represents general Database model. Singleton pattern.
 */
public class DatabaseService {
    private OracleDataSource ods;
    private Connection connection;
    private static DatabaseService instance;
    private boolean isConnected;

    static{
        try{
            instance = new DatabaseService();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Class constructor.
     */
    DatabaseService() {}

    /**
     * Get instance of class Database.
     * @return instance of Database
     */
    public static DatabaseService getInstance(){
        return instance;
    }

    /**
     * Check if database is connected to application.
     * @return true, false
     */
    private boolean isConnected() {
        return isConnected;
    }

    /**
     * Get actual database connection of class Connection.
     * @return Connection instance
     */
    public Connection getConnection() {
        return connection;
    }

    /**
     * Connects database specified by parameters to an application.
     * @param host name of Oracle server host
     * @param port integer specifying port number
     * @param serviceName name of database service
     * @return true if connection was successful, false otherwise
     */

    public boolean connectDatabase(String host, String port, String serviceName, String login, String passwd) {

        if (!this.isConnected())
        {
            try{
                String url = "jdbc:oracle:thin:@//"+host+":"+port+"/"+serviceName;
                this.ods = new OracleDataSource();
                this.ods.setURL(url);
                this.ods.setUser(login);
                this.ods.setPassword(passwd);
                try {
                    this.connection = this.ods.getConnection();
                } catch (SQLException e) {
                    return false;
                }
                this.isConnected = true;

            } catch (SQLException e) {
                e.printStackTrace();
                this.isConnected = false;
                return false;
            }
        }
        return true;
    }

    /**
     * Disconnects database from application.
     */
    public void disconnectDatabase(){
        if(this.isConnected())
        {
            try{
                this.connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
            this.isConnected = false;
        }
    }

    /**
     * Initializes database with SQL script specified by its file path.
     * @param filePath file path to SQL script
     * @throws SQLException
     */
    public void initFromSqlScript(String filePath) throws SQLException {
        String queries = "";
        try{
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            //TODO Attention! Works only for regular SQL queries with one ';' at the end (doesnt work for TRIGGERs, PROCEDUREs, and so on.)
            while( (line = reader.readLine() ) != null){
                if(!line.matches("((--)+.*)|(\\/\\**(.*|\\n*)\\**\\/)"))
                    queries += line;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (String query : Arrays.asList(queries.split(";"))) {
            try (OraclePreparedStatement preparedStatement = (OraclePreparedStatement) this.connection.prepareStatement(query)) {
                try {
                    ResultSet resultSet = preparedStatement.executeQuery();
                    resultSet.close();
                } catch (SQLException e) {
                    if(e.getMessage().contains("ORA-00942") || e.getMessage().contains("ORA-01418") ){
                        //if tries to DROP TABLE or INDEX that doesnt exist
                    }else {
                        throw e;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();

                System.out.println("*** Exception at query: " + query +"***");

            }
        }
    }
}
