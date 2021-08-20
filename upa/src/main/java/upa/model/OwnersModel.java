package upa.model;

import oracle.jdbc.OraclePreparedStatement;
import oracle.jdbc.OracleResultSet;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class OwnersModel {

    public OwnersModel(){}

    public void insertOwner(String name, String surname, String email, String telnum) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "INSERT INTO Owners (id, name, surname, email, telnum) VALUES(?,?,?,?,?)"
        );

        preparedStatement.setInt(1, dbm.getNewId("Owners"));
        preparedStatement.setString(2, name);
        preparedStatement.setString(3, surname);
        preparedStatement.setString(4, email);
        preparedStatement.setString(5, telnum);

        try {
            preparedStatement.executeUpdate();
            dbm.getService().getConnection().commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public int deleteOwner(int id) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "DELETE FROM Owners WHERE id = ?"
        );

        dbm.getService().getConnection().commit();
        preparedStatement.setInt(1, id);
        return preparedStatement.executeUpdate();
    }

    public void updateOwner(int id, String name, String surname, String email, String telnum) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "UPDATE Owners SET name=?, surname=?, email=?, telnum=? WHERE id=?"
        );

        preparedStatement.setString(1, name);
        preparedStatement.setString(2, surname);
        preparedStatement.setString(3, email);
        preparedStatement.setString(4, telnum);
        preparedStatement.setInt(5, id);
        preparedStatement.executeUpdate();
        dbm.getService().getConnection().commit();
    }

    public HashMap<String, String> getOwner(int id) throws SQLException {
        HashMap<String, String> owner = null;
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "SELECT * FROM Owners WHERE id = ?"
        );
        preparedStatement.setInt(1, id);
        OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

        if(resultSet.next()){
           owner = new HashMap<>();
           owner.put("name", resultSet.getString("name"));
           owner.put("surname", resultSet.getString("surname"));
           owner.put("email", resultSet.getString("email"));
           owner.put("telnum", resultSet.getString("telnum"));
        }

        return owner;
    }

    public HashMap<Integer, HashMap<String, String>> getAllOwners() throws  SQLException {
        HashMap<Integer, HashMap<String, String>> owners = new HashMap<>();
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "SELECT * FROM Owners"
        );

        OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

        while(resultSet.next()){
            HashMap<String, String> owner = new HashMap<>();
            owner.put("name", resultSet.getString("name"));
            owner.put("surname", resultSet.getString("surname"));
            owner.put("email", resultSet.getString("email"));
            owner.put("telnum", resultSet.getString("telnum"));
            owners.put(resultSet.getInt("id"), owner);
        }

        return owners;
    }
}
