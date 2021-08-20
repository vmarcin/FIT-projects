package upa.model;

import oracle.jdbc.OraclePreparedStatement;
import oracle.jdbc.OracleResultSet;

import java.sql.SQLException;

public class DatabaseModel {
    private DatabaseService dbs;

    public DatabaseModel(){
        this.dbs = DatabaseService.getInstance();
    }

    public DatabaseService getService(){
        return this.dbs;
    }

    /**
     * Generates id for currently inserting entry into table specified by parameter table.
     * @param table name of table existing in database
     * @return new id number
     * @throws SQLException
     */
    public int getNewId(String table) throws SQLException {
        try(OraclePreparedStatement preparedStatement= (OraclePreparedStatement) this.dbs.getConnection().prepareStatement(
                "SELECT TABLE_NAME FROM USER_TABLES")){

            try(OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery()){
                boolean isTableInDb = false;
                while(resultSet.next()) {
                    if(resultSet.getString("table_name").equalsIgnoreCase(table)){
                        isTableInDb = true;
                    }
                }

                if (!isTableInDb){
                    System.err.println("Error: getNewId(table) function call - table with name '"+table+"' doesn't exist in USER_TABLES.");
                    throw new SQLException();
                }
            }
        }

        int maxId = 0;
        try (OraclePreparedStatement preparedStatement = (OraclePreparedStatement) this.dbs.getConnection().prepareStatement(
                "SELECT MAX(id) as maxId FROM " + table)) {
            try (OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery()) {
                if (resultSet.next()) {
                    maxId = (int) resultSet.getInt("maxId");
                }
            }
        }
        return maxId+1;
    }

}

