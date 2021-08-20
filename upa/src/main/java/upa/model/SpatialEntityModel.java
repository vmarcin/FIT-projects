package upa.model;

import javafx.geometry.Point2D;
import javafx.scene.shape.Polygon;
import javafx.scene.shape.Polyline;
import oracle.jdbc.OraclePreparedStatement;
import oracle.jdbc.OracleResultSet;
import oracle.spatial.geometry.JGeometry;
import upa.controller.MapPane;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Struct;
import java.util.*;

public class SpatialEntityModel {
    public OwnersModel om;
    private static final int DIM = 2;
    private static final int SRID = 0;

    public final List<String> spatialEntityTypes = Arrays.asList("shop","school","hospital","park","block of flats", "property", "estate", "road", "tram stop", "tram line", "trees");
    public final List<String> propertyTypes = Arrays.asList("estate","house","flat");
    public final List<String> spatialEntityPolygonTypes = Arrays.asList("shop","school","hospital","park","block of flats", "property", "estate");

    private static final String SQL_SELECT_SPATIAL_ENTITIES_BY_TYPE = "SELECT id FROM SpatialEntities WHERE type=?";
    private static final String SQL_SELECT_PROPERTIES_BY_TYPE = "SELECT * FROM SpatialEntities S INNER JOIN Properties P ON S.ID = P.ID WHERE property_type=?";
    private static final String SQL_SELECT_SPATIAL_ENTITY_BY_ID = "SELECT * FROM SPATIALENTITIES WHERE id=?";
    private static final String SQL_SELECT_PROPERTY_BY_ID = "SELECT * FROM SPATIALENTITIES INNER JOIN PROPERTIES P on SPATIALENTITIES.ID = P.ID WHERE P.ID=?";
    private static final String SQL_SELECT_DISTANCE_TO_OTHERS = "SELECT a.id, a.name, b.id as bid, b.name, SDO_GEOM.SDO_DISTANCE(a.geometry, b.geometry, 0.1) AS distance FROM SpatialEntities a, SpatialEntities b WHERE a.id=? AND a.id<>b.id ORDER BY distance";
    private static final String SQL_SELECT_DISTANCE_TO_ENTITY = "SELECT a.id, a.name, b.id as bid, b.name, SDO_GEOM.SDO_DISTANCE(a.geometry, b.geometry, 0.1) AS distance FROM SpatialEntities a, SpatialEntities b WHERE a.id=? AND a.id<>b.id AND b.id=? ORDER BY distance";
    private static final String SQL_SELECT_DISTANCE_TO_ENTITIES_TYPE = "SELECT a.id, a.name, b.id as bid, b.type, b.name, SDO_GEOM.SDO_DISTANCE(a.geometry, b.geometry, 0.1) AS distance FROM SpatialEntities a, SpatialEntities b WHERE a.id=? AND a.id<>b.id AND b.type=? ORDER BY distance";
    private static final String SQL_SELECT_JOINED_ESTATES_GEOMETRY_BY_10_UNITS = "SELECT SDO_AGGR_UNION(SDOAGGRTYPE(b.geometry, 10.0))  FROM SpatialEntities a,SpatialEntities b WHERE a.id=? AND b.id in (select id from properties where property_type='estate') AND SDO_WITHIN_DISTANCE (b.geometry, a.geometry,'distance=10') = 'TRUE'";

    public SpatialEntityModel(){
        this.om = new OwnersModel();
    }

    private Boolean insertSpatialEntityJGeometry(JGeometry jGeometry, String type) {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        Boolean canInsert = false;

        try {
            if(canBeInserted(0, jGeometry, type, "")) {
                canInsert = true;
                OraclePreparedStatement preparedStatement = (OraclePreparedStatement) connection.prepareStatement(
                        "INSERT INTO SpatialEntities (id,type, geometry) VALUES (?,?,?)");
                try {
                    preparedStatement.setInt(1, dbm.getNewId("SpatialEntities"));
                    preparedStatement.setString(2, type);
                    preparedStatement.setObject(3, JGeometry.storeJS(connection, jGeometry));
                    preparedStatement.executeUpdate();
                    connection.commit();
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    preparedStatement.close();
                }
            } else {
                System.err.println("ERR: Cannot insert spatial entity! It overlaps with others.");
            }
        } catch (SQLException e) {
            System.err.println("error insert");
        }
        return canInsert;
    }

    public Boolean insertShop(Polygon polygon) {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(polygon), "shop");
    }

    public Boolean insertSchool(Polygon polygon)   {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(polygon), "school");
    }

    public Boolean insertHospital(Polygon polygon)   {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(polygon), "hospital");
    }

    public Boolean insertPark(Polygon polygon)   {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(polygon), "park");
    }

    public Boolean insertBlockOfFlats(Polygon polygon)   {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(polygon), "block of flats");
    }

    public Boolean insertTramStop(Point2D point)   {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(point), "tram stop");
    }

    public Boolean insertTramLine(Polyline polyline)   {
        return insertSpatialEntityJGeometry(createJGeometryFromObject(polyline), "tram line");
    }

    public Boolean insertTrees(Object multipoint)   {
        if (multipoint instanceof Object[] || multipoint instanceof double[]) {
            return insertSpatialEntityJGeometry(createJGeometryFromObject(multipoint), "trees");
        } else {
            //System.out.println("ERR: insertTrees: invalid Object instanceof object!");
        }
        return null;
    }

    private Boolean insertPropertyJGeometry(JGeometry jGeometry, String propertyType) {
        return insertPropertyJGeometry( jGeometry, propertyType, null);
    }

    private Boolean insertPropertyJGeometry(JGeometry jGeometry, String propertyType, MapPane map) {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        Boolean canInsert = false;
        int propertyId = 0;
        try {
            if (canBeInserted(0 ,jGeometry, "property", propertyType)) {
                canInsert = true;
                OraclePreparedStatement preparedStatement1 = (OraclePreparedStatement) connection.prepareStatement(
                        "INSERT INTO SpatialEntities (id, type, geometry) VALUES (?,?,?)");
                OraclePreparedStatement preparedStatement2 = (OraclePreparedStatement) connection.prepareStatement(
                        "INSERT INTO Properties (id, property_type) VALUES (?,?)");

                try {
                    propertyId = dbm.getNewId("SpatialEntities");

                    preparedStatement1.setInt(1, propertyId);
                    preparedStatement1.setString(2, "property");
                    preparedStatement1.setObject(3, JGeometry.storeJS(connection, jGeometry));
                    preparedStatement1.executeUpdate();

                    connection.commit();
                    preparedStatement2.setInt(1, propertyId);
                    preparedStatement2.setString(2, propertyType);
                    preparedStatement2.executeUpdate();

                    connection.commit();
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    preparedStatement1.close();
                    preparedStatement2.close();
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return canInsert;
    }

    public Boolean insertHouse(Polygon polygon)   {
        return insertPropertyJGeometry(createJGeometryFromObject(polygon), "house");
    }

    public Boolean insertEstate(Polygon polygon, MapPane map)   {
        return insertPropertyJGeometry(createJGeometryFromObject(polygon), "estate", map);
    }

    public Boolean insertFlat(Polygon polygon)   {
        return insertPropertyJGeometry(createJGeometryFromObject(polygon), "flat");
    }

    public void joinEstatesBackend(Integer propertyId, MapPane map){
            DatabaseModel dbm = new DatabaseModel();
            Connection connection = dbm.getService().getConnection();
            try {
                OraclePreparedStatement preparedStatementEstatesJoin = (OraclePreparedStatement) connection.prepareStatement(
                        SQL_SELECT_JOINED_ESTATES_GEOMETRY_BY_10_UNITS);

                preparedStatementEstatesJoin.setInt(1, propertyId);
                OracleResultSet resultSet = (OracleResultSet) preparedStatementEstatesJoin.executeQuery();

                JGeometry newJoinedEstateJGeometry = null;
                if (resultSet.next()) {
                    Struct obj = (Struct) resultSet.getObject(1);
                    newJoinedEstateJGeometry = JGeometry.loadJS(obj);
//                    //System.out.println("<<<<<<<<<<<<<<< " + createObjectFromJGeometry(newJoinedEstateJGeometry) + "<<<<<<<<<<<<<<<<<<<<<<<<");
                    HashMap<Integer, String> relationShipsToJoinedGeometry = getRelationShips(newJoinedEstateJGeometry);
                    Set<Integer> estatesUnderIds = new HashSet<>(getAllEstatesAsSpatialEntities());

                    while (relationShipsToJoinedGeometry.values().remove("DISJOINT")) ;
                    relationShipsToJoinedGeometry.keySet().removeAll(estatesUnderIds);
                    relationShipsToJoinedGeometry.remove(propertyId);

                    if (createObjectFromJGeometry(newJoinedEstateJGeometry) instanceof Polygon) {
                        List<Integer> propertyEstatesIds = getAllEstates();
                        Struct newJoinedEstate = JGeometry.storeJS(connection, newJoinedEstateJGeometry);
                        OraclePreparedStatement preparedStatementGetOldEstateId = (OraclePreparedStatement) connection.prepareStatement(
                                "SELECT id  FROM SpatialEntities WHERE SDO_RELATE(?,?,'MASK=anyinteract') = 'TRUE'");
                        List<Integer> relatingIds = new ArrayList<>();
                        preparedStatementGetOldEstateId.setObject(1, JGeometry.storeJS(connection, newJoinedEstateJGeometry));
                        for (Integer EstateId : propertyEstatesIds) {
                            preparedStatementGetOldEstateId.setObject(2, JGeometry.storeJS(connection, getJGeometry(EstateId)));
                            OracleResultSet resultSetNewJoinedEstateRelatingIds = (OracleResultSet) preparedStatementGetOldEstateId.executeQuery();

                            while (resultSetNewJoinedEstateRelatingIds.next()) {
                                int relId = resultSetNewJoinedEstateRelatingIds.getInt("id");
                                if (relationShipsToJoinedGeometry.containsKey(relId) && !relatingIds.contains(relId)) {
                                    relatingIds.add(relId);
                                }
                            }
                            resultSetNewJoinedEstateRelatingIds.close();
                        }

                        preparedStatementGetOldEstateId.close();

//                        //System.out.println("<<< all estates ids (in database): <<< " + propertyEstatesIds);
//                        //System.out.println("<<< ids of objects that relate to new polygon: <<< " + relatingIds);
                        for (Integer id : relatingIds) {
                            deleteProperty(id);
                        }
                        updateJGeometry(propertyId, newJoinedEstateJGeometry, "");

                        relatingIds.add(propertyId);

                        if (map != null) {
                            map.joinEstates(relatingIds, propertyId);
                        }

                    } else {
//                        //System.out.println("neni polygon");
//                        //System.out.println(newJoinedEstateJGeometry);
//                        //System.out.println(createObjectFromJGeometry(newJoinedEstateJGeometry));
                    }
                }
            } catch (SQLException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
    }



    public boolean canBeUpdated(int id, JGeometry jGeometry, String type, String propertyType) throws SQLException {
        return canBeInserted(id, jGeometry, type, propertyType);
    }

    public Boolean updateJGeometry(int id, JGeometry newJGeometry, String type) throws Exception {
        Boolean canUpdate = false;
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) connection.prepareStatement(
                "UPDATE SpatialEntities SET geometry=? WHERE id=?"
        );
        if(canBeUpdated(id, newJGeometry, type, getPropertyType(id))) {
            canUpdate = true;
            try {
                preparedStatement.setObject(1, JGeometry.storeJS(connection, newJGeometry));
                preparedStatement.setInt(2, id);
                preparedStatement.executeUpdate();
                connection.commit();
            } catch(Exception e) {}
            finally {
                preparedStatement.close();
            }
        }
        return canUpdate;
    }

    public Boolean updateJGeometry(int id, Object object, String type) {
        try {
            return updateJGeometry(id, createJGeometryFromObject(object), type);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public String getType(int id) {
        DatabaseModel dbm = new DatabaseModel();
        try{
            OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                    "SELECT type FROM SpatialEntities WHERE id=?"
            );
            try {
                preparedStatement.setInt(1, id);
                OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

                if(resultSet.next()){
                    return resultSet.getString("type");
                }else{
                    System.err.println("doesnt exist id " + id);
                }

                resultSet.close();
            } catch (SQLException e) {
                e.printStackTrace();
            } finally {
                preparedStatement.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return "";
    }

    public String getPropertyType(int id) {
        DatabaseModel dbm = new DatabaseModel();
        try{
            OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                    "SELECT property_type FROM Properties WHERE id=?"
            );
            try {
                preparedStatement.setInt(1, id);
                OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

                if(resultSet.next()){
                    return resultSet.getString("property_type");
                }
//                else{
//                    System.err.println("getPropertyType: doesnt exist id " + id);
//                }
                resultSet.close();

            } catch (SQLException e) {
                e.printStackTrace();
            } finally {
                preparedStatement.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return "";
    }

    public boolean isBlockEmpty(int blockId){
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        List<Integer> flatsIds = new ArrayList<>();
        try {
            OraclePreparedStatement preparedStatement = (OraclePreparedStatement) connection.prepareStatement(
                    "SELECT id  FROM SpatialEntities WHERE SDO_RELATE(?,geometry,'MASK=CONTAINS') = 'TRUE'"
            );
            preparedStatement.setObject(1, JGeometry.storeJS(connection, getJGeometry(blockId)));
            ResultSet resultSet = preparedStatement.executeQuery();

            while (resultSet.next()) {
                flatsIds.add(resultSet.getInt("id"));
            }

            preparedStatement.close();
            resultSet.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        //System.out.println(flatsIds);

        if (flatsIds.isEmpty())
            return true;
        else
            return false;
    }

    public int deleteProperty(int id) {
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = null;
        try {
            preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                    "DELETE FROM Properties WHERE id = ?"
            );
        } catch (SQLException e) {
            e.printStackTrace();
        }
        try {
            preparedStatement.setInt(1, id);
            preparedStatement.executeUpdate();
            dbm.getService().getConnection().commit();
            preparedStatement.close();
            return deleteSpatialEntity(id);
        } catch (SQLException e) {
//            System.err.println("<<<<<<<<<<<< ERR: deleteSpatialEntity: spatialEntity with id:"+ id + "doesnt exist");
            return -1;
        }
    }

    public int deleteSpatialEntity(int id) {
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = null;
        try {
            preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                    "DELETE FROM SpatialEntities WHERE id = ?"
            );
        } catch (SQLException e) {
            e.printStackTrace();
        }
        try {
            preparedStatement.setInt(1, id);
            preparedStatement.executeUpdate();
            dbm.getService().getConnection().commit();
            preparedStatement.close();
            return id;
        } catch (SQLException e) {
//            System.err.println("<<<<<<<<<<<< ERR: deleteSpatialEntity: spatialEntity with id:"+ id + "doesnt exist");
            return -1;
        }
    }

    //<id, relationship> to actually inserting object
    public HashMap<Integer, String> getRelationShips(JGeometry jGeometry) throws SQLException {
        HashMap<Integer, String> relationShips = new HashMap<>();
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        OraclePreparedStatement preparedStatementRelations = (OraclePreparedStatement) connection.prepareStatement(
                "SELECT id, SDO_GEOM.RELATE(geometry, 'DETERMINE', ?, 0.1) relationship FROM SpatialEntities"
        );
        try {
            preparedStatementRelations.setObject(1, JGeometry.storeJS(connection, jGeometry));
            OracleResultSet resultSet = (OracleResultSet) preparedStatementRelations.executeQuery();
            while (resultSet.next()) {
//                //System.out.println("id: " + resultSet.getString("id") + "| relation: " + resultSet.getString("relationship"));
                relationShips.put(resultSet.getInt("id"), resultSet.getString("relationship"));
            }
            resultSet.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("ERR: insertSpatialEntityWithJGeometry: SELECT: executeQuery");
        } finally {
            preparedStatementRelations.close();
        }

        return  relationShips;
    }

    public boolean canBeInserted(int id, JGeometry jGeometry, String type, String propertyType) throws SQLException {
        boolean canBeInserted = false;
        if (type.equals("tram stop")){

            HashMap<Integer, String> relationShips = getRelationShips(jGeometry);
//            //System.out.println("realtionShips of Point with other geometries !!!!:   "+ relationShips.toString());

            Integer roadId = getSpatialEntitiesByType("road").get(0);
            List<Integer> tramLines = getSpatialEntitiesByType("tram line");

            Point2D point = (Point2D) createObjectFromJGeometry(jGeometry);
            boolean tramLinesRelatingToTramStopCircle = tramStopCircleRelatesTramLines(point.getX(), point.getY(), 7);

            if (relationShips.get(roadId).equals("CONTAINS") && tramLinesRelatingToTramStopCircle){
                canBeInserted = true;
            }
        } else if (type.equals("tram line")){
//            //System.out.println("tram line");
            HashMap<Integer, String> relationShips = getRelationShips(jGeometry);
            Integer roadId = getSpatialEntitiesByType("road").get(0);

//            //System.out.println("roadId: " + roadId);
//            //System.out.println(relationShips.toString());

            if (relationShips.get(roadId).equals("COVERS") || relationShips.get(roadId).equals("CONTAINS")){
                canBeInserted = true;
            }

        } else {
            HashMap<Integer, String> relationShips = getRelationShips(jGeometry);
            List<Integer> estates = getSpatialEntitiesByType("estate");
            Integer estateKey = 0;


            relationShips.remove(id);

            for (Integer key: relationShips.keySet()) {
                if (relationShips.get(key).equals("COVERS") || relationShips.get(key).equals("CONTAINS")){
                    if(estates.contains(key)) {
                        estateKey = key;
                        canBeInserted = true;
                    }
                }
            }

            if(type.equals("property") && propertyType.equals("flat")) {
                List<Integer> blockOfFlatsIds = getAllBlockOfFlats();
                Integer blockOfFlatKey = 0;

                for (Integer key: relationShips.keySet()) {
                    if (relationShips.get(key).equals("COVERS") || relationShips.get(key).equals("CONTAINS")){
                        if(blockOfFlatsIds.contains(key)) {
                            blockOfFlatKey = key;
                            canBeInserted = true;
                        }
                    }
                }

                if(!canBeInserted || blockOfFlatKey == 0)
                    return false;
                if(relationShips.get(blockOfFlatKey) != null)
                    relationShips.remove(blockOfFlatKey);
                if(relationShips.get(estateKey) != null)
                    relationShips.remove(estateKey);
                for (Integer key: relationShips.keySet()) {
                    if(!relationShips.get(key).equals("DISJOINT")){
                        return false;
                    }
                }
            }
            if(!canBeInserted)
                return false;

            if(relationShips.get(estateKey) != null)
                relationShips.remove(estateKey);

            relationShips.remove(26);

            for (Integer key: relationShips.keySet()) {
                if(!relationShips.get(key).equals("DISJOINT")){
//                    //System.out.println(key);
                    canBeInserted = false;
                }
            }
        }
//        //System.out.println("can " + canBeInserted);
        return canBeInserted;
    }

    public boolean isProperty(int id) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "SELECT id FROM Properties WHERE id=?"
        );
        try {
            preparedStatement.setInt(1, id);
            int result = preparedStatement.executeUpdate();
            return result==1;

        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }
        return false;
    }

    public String createSqlUpdateSetSequence(HashMap<String, Object> info){
        String setSequence = "";
        int commaCounter = info.keySet().size();
        for (String key: info.keySet()) {
            if (commaCounter > 1) {
                setSequence += (key + "=?, ");
            } else {
                setSequence += (key + "=? ");
            }
            commaCounter--;
        }
        return setSequence;
    }

    public void updateInfoSpatialEntity(int id, Map<String, String> infoSpatialEntity) throws SQLException {
        HashMap<String, Object> infoHash = new HashMap<>(infoSpatialEntity);
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) connection.prepareStatement(
                "UPDATE SpatialEntities SET " + createSqlUpdateSetSequence(infoHash) + " WHERE id=?"
        );


        try {
            int indexCounter = 1;
            for (String key:infoHash.keySet()) {
                preparedStatement.setString(indexCounter++, (String) infoHash.get(key));
            }
            preparedStatement.setInt(indexCounter, id);
            preparedStatement.executeUpdate();
            connection.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }
    }

    public void updateInfoProperty(int id, Map<String, Object> infoProperty) throws SQLException {
        HashMap<String, Object> infoHash = new HashMap<>(infoProperty);
        DatabaseModel dbm = new DatabaseModel();
        OwnersModel om = new OwnersModel();
        Connection connection = dbm.getService().getConnection();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) connection.prepareStatement(
                "UPDATE Properties SET " + createSqlUpdateSetSequence(infoHash) + " WHERE id=?"
        );

        try {
            int indexCounter = 1;
            for (String key:infoHash.keySet()) {
                Object obj = infoHash.get(key);
                if (obj instanceof String){
                    //System.out.println(key);
                    preparedStatement.setString(indexCounter++, (String) infoHash.get(key));
                } else if (obj instanceof Integer) {
                    //System.out.println(key);
                    if (om.getOwner((Integer) infoHash.get(key)) == null) {
                        throw new Exception();
                    }
                    preparedStatement.setInt(indexCounter++, (Integer) infoHash.get(key));
                } else {
                    //System.out.println("ERR: Unknown instancof type!");
                    throw new Exception();
                }
            }
            preparedStatement.setInt(indexCounter, id);
            preparedStatement.executeUpdate();
            connection.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("ERR: updateInfoProperty:  Owner with id: "+ infoHash.get("id_owner") +" doesnt exist!");
        } finally {
            preparedStatement.close();
        }
    }

    public Object[] createDoubleCouples(double[] points){
        Object[] doubleCouples = new Object[points.length/2];
        for (int i=0, j=0; i < points.length; i+=2, j++) {
            doubleCouples[j] = new double[]{points[i],points[i+1]};
        }
        return doubleCouples;
    }

    public Polygon createFivePointsPolygon(Polygon polygon) {
        Polygon fivePointsPolygon = new Polygon();
        double[] points = getPrimitiveDoubleArray(polygon.getPoints());
        //Rectangle POLYGONS consisting from 2 points
        if (points.length == 4) {
            double x1 = points[0];
            double y1 = points[1];

            double x2 = points[2];
            double y2 = points[3];

            fivePointsPolygon.getPoints().addAll(x1,y1, x2,y1, x2,y2, x1,y2, x1,y1);
        }else{
            return polygon;
        }
        return fivePointsPolygon;
    }
    
    public JGeometry createJGeometryFromObject(Object object) {
        JGeometry jGeometry = null;
        if (object instanceof Polygon){
            jGeometry = JGeometry.createLinearPolygon(getPrimitiveDoubleArray(((Polygon) object).getPoints()), 2,0);
        }else if (object instanceof Polyline){
            jGeometry = JGeometry.createLinearLineString(getPrimitiveDoubleArray(((Polyline) object).getPoints()),2,0);
        }else if (object instanceof Point2D){
            jGeometry = JGeometry.createPoint(new double[]{((Point2D) object).getX(), ((Point2D) object).getY()}, DIM, SRID);
        }else if (object instanceof Object[]){
            jGeometry = JGeometry.createMultiPoint((Object[]) object, 2, 0);
        }else if (object instanceof double[]) {
            jGeometry = JGeometry.createMultiPoint( createDoubleCouples((double[]) object), 2, 0);
        }else{
            System.err.println("ERR: createJGeometryFromObject: Unknown Object type!");
        }
        return jGeometry;
    }

    public Object createObjectFromJGeometry(JGeometry jGeometry) {

        double[] ordArray;
        switch (jGeometry.getType()) {
            case JGeometry.GTYPE_POLYGON:
                return createFivePointsPolygon(new Polygon(jGeometry.getOrdinatesArray()));

            case JGeometry.GTYPE_CURVE:
                return new Polyline(jGeometry.getOrdinatesArray());

            case JGeometry.GTYPE_POINT:
                ordArray = jGeometry.getPoint();
                return new Point2D(ordArray[0], ordArray[1]);

            case JGeometry.GTYPE_MULTIPOINT:
                List<Point2D> points = new ArrayList<>();
                ordArray = jGeometry.getOrdinatesArray();
                for (int i = 0; i < ordArray.length; i+=2) {
                    double x = ordArray[i];
                    double y = ordArray[i+1];
                    points.add(new Point2D(x,y));
                }
                return points;

            default:
                //System.out.println("Type: " + jGeometry.getType());
                return null;
        }
    }

    public double[] getPrimitiveDoubleArray(List<Double> doubles) {
        double[] points = new double[doubles.size()];
        for (int i = 0; i < points.length ; i++) {
            points[i] = doubles.get(i);
        }
        return points;
    }

    public boolean checkJGeometryValidity() {
        DatabaseModel dbm = new DatabaseModel();
        boolean invalidGeometry = false;
        try {
            OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                    "SELECT name, id, SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(geometry, 0.1) valid FROM SpatialEntities"
            );
            OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

            while (resultSet.next()) {

                if (!resultSet.getString("valid").equals("TRUE")) {

//                if (!invalidGeometry){
//                    //System.out.println("<<< checkJGeometryValidity: Invalid geometries:");
//                }
                    invalidGeometry = true;
//                //System.out.println("id: " + resultSet.getInt("id") + " " + resultSet.getString("valid"));
                }
            }
            preparedStatement.close();
            resultSet.close();
        } catch (SQLException e) {}

        if (!invalidGeometry){
//            //System.out.println("<<< checkJGeometryValidity: All geometries are VALID!");
            return true;
        }
        return false;
    }

    public List<Integer> getSpatialEntityAtCoors(double x, double y) throws  SQLException {
        List<Integer> spatialEntitiesIds = new ArrayList<>();
        DatabaseModel dbm = new DatabaseModel();

        double r = 4; //more likely to hit something

        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
        "SELECT id FROM SpatialEntities WHERE SDO_RELATE(geometry, SDO_GEOMETRY(2003, NULL, NULL, SDO_ELEM_INFO_ARRAY(1,1003,4), SDO_ORDINATE_ARRAY(?,?,?,?,?,?)), 'mask=ANYINTERACT') = 'TRUE'"
        );
        preparedStatement.setDouble(1, x);
        preparedStatement.setDouble(2,y-r);
        preparedStatement.setDouble(3, x+r);
        preparedStatement.setDouble(4,y);
        preparedStatement.setDouble(5,x);
        preparedStatement.setDouble(6,y+r);

        try {
            OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

            try{
                while (resultSet.next()) {
                    spatialEntitiesIds.add(resultSet.getInt("id"));
                }
            } finally {
                resultSet.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();

        }
        return spatialEntitiesIds;
    }

    public HashMap<String, String> getSpatialEntityInfo(int id, String select) throws SQLException {
        HashMap<String, String> entity = new HashMap<>();
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(select);
        preparedStatement.setInt(1, id);

        try{
            ResultSet resultSet = preparedStatement.executeQuery();
            if(resultSet.next()){
                entity.put("name", resultSet.getString("name"));
                entity.put("type", resultSet.getString("type"));
                entity.put("description", resultSet.getString("description"));

                if(select.equals(SQL_SELECT_PROPERTY_BY_ID)){
                    entity.put("address", resultSet.getString("address"));
                    entity.put("price", resultSet.getString("price"));
                    entity.put("property_type", resultSet.getString("property_type"));
                    entity.put("id_owner", resultSet.getString("id_owner"));

                    entity.put("area", String.valueOf(getArea(id)));
                    entity.put("length", String.valueOf(getLength(id)));
                }
            } else {
                if(select.equals(SQL_SELECT_PROPERTY_BY_ID)){
                    System.err.println("ERR: No such property with id: "+id);
                }else{
                    System.err.println("ERR: No such spatial entity with id: "+id);
                }
                throw new Exception();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }
        return entity;
    };

    public HashMap<String, String> getSpatialEntityInfo(int id) throws SQLException {
       return getSpatialEntityInfo(id, SQL_SELECT_SPATIAL_ENTITY_BY_ID);
    }

    public HashMap<String, String> getPropertyInfo(int id) throws SQLException {
        return getSpatialEntityInfo(id, SQL_SELECT_PROPERTY_BY_ID);
    }

    public JGeometry getJGeometry(int id) {
        JGeometry jGeometry = null;
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = null;
        try {
            preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                    "SELECT id, geometry FROM SpatialEntities WHERE id=?"
            );
        } catch (SQLException e) {
            System.err.println("getJgoemetry prepareStatement error!");
        }
        try{
            preparedStatement.setInt(1, id);
            ResultSet resultSet = preparedStatement.executeQuery();

            if(resultSet.next()){
                Struct obj = (Struct) resultSet.getObject("geometry");
                jGeometry = JGeometry.loadJS(obj);

            } else {
                System.err.println("ERR: No spatial entity polygon with id: "+id);
            }
            preparedStatement.close();
            resultSet.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return jGeometry;
    }

    public boolean tramStopCircleRelatesTramLines(double x, double y, double r) throws SQLException {
        List<Integer> tramLinesRelatesIds = new ArrayList<>();
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        List<Integer> tramLinesIds = getAllTramLines();
        //System.out.println("getAllTramLines():  " + tramLinesIds);
        OraclePreparedStatement preparedStatementRelations = (OraclePreparedStatement) connection.prepareStatement(
                "SELECT id  FROM SpatialEntities WHERE SDO_RELATE(?,?,'MASK=anyinteract') = 'TRUE'"
        );

        JGeometry circle = JGeometry.createCircle(x, y, r, SRID);
        boolean circleIntersectsTramLine = false;

        for (Integer tramLineId : tramLinesIds) {
            JGeometry tramLineJGeometry = getJGeometry(tramLineId);
            try {
                preparedStatementRelations.setObject(1, JGeometry.storeJS(connection, circle));
                preparedStatementRelations.setObject(2, JGeometry.storeJS(connection, tramLineJGeometry));
            } catch (Exception e) {
                e.printStackTrace();
            }

            OracleResultSet resultSet = (OracleResultSet) preparedStatementRelations.executeQuery();

//            //System.out.println("TramLineID="+tramLineId);
            while (resultSet.next()) {
//                //System.out.println(resultSet.getInt("id"));
                if(resultSet.getInt("id") == tramLineId){
                    circleIntersectsTramLine = true;
                    break;
                }
//                //System.out.println("id: "+ resultSet.getString("id"));
                preparedStatementRelations.close();
                resultSet.close();
            }
        }
        return circleIntersectsTramLine;
    }

    /**
     *
     * @param id
     * @return
     * @throws SQLException
     */
    public Object getJavaFxObject(int id) throws SQLException {
        Object javaFxObject = null;
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "SELECT id, geometry FROM SpatialEntities WHERE id=?"
        );
        preparedStatement.setInt(1, id);

        try {
            ResultSet resultSet = preparedStatement.executeQuery();
            try {
                if(resultSet.next()){
                    Struct obj = (Struct) resultSet.getObject(2);
                    JGeometry jgeom = JGeometry.loadJS(obj);
                    javaFxObject = createObjectFromJGeometry(jgeom);

                } else {
                    System.err.println("ERR: No spatial entity polygon with id: "+id);
                    throw new Exception();
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                resultSet.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }
        return javaFxObject;
    }


    private List<Integer> getSpatialEntitiesByType(String spatialEntityType, String select) throws SQLException {
        List<Integer> entitiesIds = new ArrayList<>();
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(select);
        preparedStatement.setString(1, spatialEntityType);

        try {
            ResultSet resultSet = preparedStatement.executeQuery();
            try {
                while(resultSet.next()){
                    entitiesIds.add(resultSet.getInt("id"));
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                resultSet.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }
        return entitiesIds;
    }

    public List<Integer> getSpatialEntitiesByType(String spatialEntityType) {
        try {
            return getSpatialEntitiesByType(spatialEntityType, SQL_SELECT_SPATIAL_ENTITIES_BY_TYPE);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return new ArrayList<>();
    }

    public List<Integer> getAllEstatesAsSpatialEntities() {
        return getSpatialEntitiesByType("estate");
    }
    public List<Integer> getAllShops() {
        return getSpatialEntitiesByType("shop");
    }

    public List<Integer> getAllSchools() {
        return getSpatialEntitiesByType("school");
    }

    public List<Integer> getAllTramStops() {
        return getSpatialEntitiesByType("tram stop");
    }

    public List<Integer> getAllTramLines() {
        return getSpatialEntitiesByType("tram line");
    }

    public List<Integer> getAllHospitals() {
        return getSpatialEntitiesByType("hospital");
    }

    public List<Integer> getAllTrees() {
        return getSpatialEntitiesByType("trees");
    }

    public List<Integer> getAllParks() {
        return getSpatialEntitiesByType("park");
    }

    public List<Integer> getAllProperties() {
        return getSpatialEntitiesByType("property");
    }

    public List<Integer> getAllRoads() {
        return getSpatialEntitiesByType("road");
    }

    public List<Integer> getAllBlockOfFlats() {
        return getSpatialEntitiesByType("block of flats");
    }

    public List<Integer> getPropertiesByType(String propertyType) {
        try {
            return getSpatialEntitiesByType(propertyType, SQL_SELECT_PROPERTIES_BY_TYPE);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return new ArrayList<>();
    }

    public List<Integer> getAllFlats() {
        return getPropertiesByType("flat");
    }

    public List<Integer> getAllHouses() {
        return getPropertiesByType("house");
    }

    public List<Integer> getAllEstates() {
        return getPropertiesByType("estate");
    }


    /*
        SPATIAL OPERATIONS
     */

    public double getArea(int id) throws SQLException {
        double area = 0.0;
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "SELECT id, SDO_GEOM.SDO_AREA(geometry, 0.1) area FROM SpatialEntities WHERE id=?"
        );

        preparedStatement.setInt(1, id);

        try {
            ResultSet resultSet = preparedStatement.executeQuery();
            try {
                if(resultSet.next()){
                    area = resultSet.getDouble(2);
                } else {
                    System.err.println("ERR: No spatial entity polygon with id: "+id);
                    throw new Exception();
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                resultSet.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }

        return area;
    }

    public double getLength(int id) throws SQLException {
        double length = 0.0;
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(
                "SELECT id, SDO_GEOM.SDO_LENGTH(geometry, 0.1) length FROM SpatialEntities WHERE id=?"
        );

        preparedStatement.setInt(1, id);

        try {
            ResultSet resultSet = preparedStatement.executeQuery();
            try {
                if(resultSet.next()){
                    length = resultSet.getDouble(2);
                } else {
                    System.err.println("ERR: No spatial entity polygon with id: "+id);
                    throw new Exception();
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                resultSet.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }

        return length;
    }

    HashMap<Integer, Double> getDistancesToEntities(int id, int idOtherEntity, String type, String select) throws SQLException {
        HashMap<Integer, Double> distances = new LinkedHashMap<>();
        DatabaseModel dbm = new DatabaseModel();
        OraclePreparedStatement preparedStatement = (OraclePreparedStatement) dbm.getService().getConnection().prepareStatement(select);

        preparedStatement.setInt(1, id);
        if (select.equals(SQL_SELECT_DISTANCE_TO_ENTITY)){
            preparedStatement.setInt(2, idOtherEntity);
        } else if (select.equals(SQL_SELECT_DISTANCE_TO_ENTITIES_TYPE)){
            preparedStatement.setString(2, type);
        }
        try {
            ResultSet resultSet = preparedStatement.executeQuery();
            try {
                while(resultSet.next()){
                   distances.put(resultSet.getInt("bid"), resultSet.getDouble("distance"));
                }
            } finally {
                resultSet.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            preparedStatement.close();
        }
        return distances;
    }

    public HashMap<Integer, Double> getDistancesToAllEntities(int id) throws SQLException {
        return getDistancesToEntities(id, 0, "", SQL_SELECT_DISTANCE_TO_OTHERS);
    }

    public HashMap<Integer, Double> getDistanceToEntity(int id, int idOtherEntity) throws SQLException {
        return getDistancesToEntities(id, idOtherEntity, "", SQL_SELECT_DISTANCE_TO_ENTITY);
    }

    public HashMap<Integer, Double> getDistanceToEntitiesType(int id, String type) throws SQLException {
        return getDistancesToEntities(id, 0, type, SQL_SELECT_DISTANCE_TO_ENTITIES_TYPE);
    }

    public HashMap<Integer, Double> getNNearestNeighboursByType(int id, int n, String type){
        HashMap<Integer, Double> distances = new LinkedHashMap<>();
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        try {
            OraclePreparedStatement preparedStatement = (OraclePreparedStatement) connection.prepareStatement(
                    "SELECT b.id bid, SDO_NN_DISTANCE(1) distance \n" +
                            "FROM SpatialEntities a, SpatialEntities b\n" +
                            "WHERE SDO_NN(b.geometry, a.geometry, 'sdo_batch_size=10', 1) = 'TRUE' AND a.id<>b.id AND a.id=? AND b.type=? AND ROWNUM<=? ORDER BY distance");

            preparedStatement.setInt(1, id);
            preparedStatement.setString(2, type);
            preparedStatement.setInt(3, n);

            OracleResultSet resultSet = (OracleResultSet) preparedStatement.executeQuery();

            while (resultSet.next()){
                distances.put(resultSet.getInt("bid"), resultSet.getDouble("distance"));
            }

            preparedStatement.close();
            resultSet.close();

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return distances;
    }
}



