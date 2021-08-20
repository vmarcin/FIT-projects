package upa.model.multimedia;

import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import oracle.jdbc.OraclePreparedStatement;
import oracle.jdbc.OracleResultSet;
import oracle.ord.im.OrdImage;
import upa.model.DatabaseModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Picture {
    private static final String SQL_INSERT_EMPTY = "INSERT INTO pictures (id, image, id_spatial_entity, is_title) VALUES (?, ordsys.ordimage.init(), ?, ?)";
    private static final String SQL_UPDATE_IMAGE = "UPDATE pictures SET image = ? WHERE id = ?";
    private static final String SQL_SELECT_FOR_UPDATE = "SELECT image FROM pictures WHERE id = ? FOR UPDATE";
    private static final String SQL_UPDATE_STILLIMAGE = "UPDATE pictures p SET p.image_si = SI_StillImage(p.image.getContent()) WHERE p.id = ?";
    private static final String SQL_UPDATE_STILLIMAGE_DATA = "UPDATE pictures SET image_ac = SI_AverageColor(image_si), image_ch = SI_ColorHistogram(image_si), image_pc = SI_PositionalColor(image_si), image_tx = SI_Texture(image_si) WHERE id = ?";
    private static final String SQL_DELETE = "DELETE FROM pictures WHERE id = ?";
    private static final String SQL_DELETE_IMAGES = "DELETE FROM pictures WHERE id_spatial_entity = ?";
    private static final String SQL_SIMILAR_IMAGE = "SELECT dest.id_spatial_entity, dest.id, SI_ScoreByFtrList(new SI_FeatureList(pic.image_ac, ?,pic.image_ch, ?,pic.image_pc, ?,pic.image_tx, ?),dest.image_si) AS similarity FROM pictures pic INNER JOIN properties prop ON pic.id_spatial_entity = prop.id, pictures dest INNER JOIN properties prop2 ON dest.id_spatial_entity = prop2.id WHERE (pic.id = (SELECT id FROM pictures WHERE id_spatial_entity = ? AND is_title = 1)) AND (dest.is_title = 1)  AND (pic.id <> dest.id) AND (pic.id_spatial_entity <> dest.id_spatial_entity) ORDER BY similarity ASC";
    private static final String SQL_SELECT_IMAGE_IMAGE_ID_TO_SPATIAL_ENTITY = "SELECT id, image FROM pictures WHERE id_spatial_entity = ?";
    private static final String SQL_IMAGES_COUNT = "SELECT COUNT(*) FROM pictures WHERE id_spatial_entity = ? AND is_title = 1";

    /**
     * Construct a new Picture.
     */
    public Picture() {}

    /**
     * Get OrdImage from database according to id of saved picture.
     * @param id id of the picture from database
     * @return ordImage for update
     * @throws SQLException
     */
    private OrdImage getOrdImageForUpdate(int id) throws SQLException{
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();

        OrdImage ordImage = null;
        try (PreparedStatement preparedStatementSelectForUpdate = connection.prepareStatement(SQL_SELECT_FOR_UPDATE)) {
            preparedStatementSelectForUpdate.setInt(1, id);
            try (ResultSet resultSet = preparedStatementSelectForUpdate.executeQuery()) {
                if (resultSet.next()) {
                    final OracleResultSet oracleResultSet = (OracleResultSet) resultSet;
                    ordImage = (OrdImage) oracleResultSet.getORAData("image", OrdImage.getORADataFactory());
                }
            }
        }
        return ordImage;
    }

    /**
     * Renew information about Still Image in the database after the OrdImage has changed.
     * @param id id of the picture in database
     * @throws SQLException
     */
    private void restoreStillImage(int id) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();

        try (PreparedStatement preparedStatementUpdateStillImage = connection.prepareStatement(SQL_UPDATE_STILLIMAGE)) {
            preparedStatementUpdateStillImage.setInt(1, id);
            preparedStatementUpdateStillImage.executeUpdate();
        }
        try (PreparedStatement preparedStatementUpdateStillImageData = connection.prepareStatement(SQL_UPDATE_STILLIMAGE_DATA)) {
            preparedStatementUpdateStillImageData.setInt(1, id);
            preparedStatementUpdateStillImageData.executeUpdate();
        }
    }

    /**
     * Insert record with empty OrdImage to database.
     * @param id id of picture in database
     * @param spatialEntityId if of property related to picture
     * @throws SQLException
     */
    private void insertEmptyOrdImage(int id, int spatialEntityId) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        int isTitle = 0;
        int count = -1;

        //find how many pictures of spatial entity marked as title are in database and set flag is_title according to it (1 if property does not have title picture)
        try(PreparedStatement preparedStatementPicturesCount = connection.prepareStatement(SQL_IMAGES_COUNT)) {
            preparedStatementPicturesCount.setInt(1, spatialEntityId);
            try (ResultSet resultSet = preparedStatementPicturesCount.executeQuery()) {
                if (resultSet.next()) {
                  count = resultSet.getInt("count(*)");
                    if (count == 0) {
                        isTitle = 1;
                    }
                }
            }
        }

        //insert new record with an empty image
        try (PreparedStatement preparedStatementInsertNew = connection.prepareStatement(SQL_INSERT_EMPTY)) {
            preparedStatementInsertNew.setInt(1, id);
            preparedStatementInsertNew.setInt(2, spatialEntityId);
            preparedStatementInsertNew.setInt(3, isTitle);
            preparedStatementInsertNew.executeUpdate();
        }
    }

    /**
     * Insert picture from file identified with filename and id of spatial entity to database.
     * @param filename file with image to insert
     * @paaram spatialEntityId id of the spatial entity related to picture
     * @throws SQLException
     * @throws IOException
     */
    public void insertOrdImageFromFile(String filename, int spatialEntityId) throws SQLException, IOException {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        final int id = dbm.getNewId("pictures");
        final boolean currentAutoCommit = connection.getAutoCommit();
        connection.setAutoCommit(false);
        insertEmptyOrdImage(id, spatialEntityId);
        try {
            //gain previously stored empty image for updating
             OrdImage imageProxy = getOrdImageForUpdate(id);
            //load picture from file
            imageProxy.loadDataFromFile(filename);
            //set properties of Oracle object from Java object
            imageProxy.setProperties();
            //update empty image by the image gained from file
            try (PreparedStatement preparedStatementUpdateImage = connection.prepareStatement(SQL_UPDATE_IMAGE)) {
                final OraclePreparedStatement oraclePreparedStatement = (OraclePreparedStatement) preparedStatementUpdateImage;
                oraclePreparedStatement.setORAData(1, imageProxy);
                preparedStatementUpdateImage.setInt(2, id);
                preparedStatementUpdateImage.executeUpdate();
            }
        } finally {
            //renew StillImage and information about it
           restoreStillImage(id);
           connection.commit();
           connection.setAutoCommit(currentAutoCommit);

        }
    }

    /**
     * Save initial data to database.
     */
    public void initPictures () {
        //insert images from files into database
        for (int i = 1; i <=33; i++) {
            Picture picture =  new Picture();

            try {
               if (i == 2){
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 45);
                }
               else if (i == 4){
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 48);
                }
               else if (i == 5){
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 49);
                }
               else if (i == 6){
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 50);
                }
               else if (i >= 14 && i <= 15) {
                   insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 46);
               }
                else if (i == 8){
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 46);
                }
               else if (i >= 22 && i <= 23) {
                   insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 45);
               }
                else if (i == 26) {//tesco
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 27);
                }
                else if (i == 27) {//billa
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 28);
                }
                else if (i == 28) {//lidl
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 29);
                }
                else if (i == 29) {//pko
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 30);
                }
                else if (i == 30) {//skola
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 32);
                }
                else if (i == 31) {//gymnazium
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 31);
                }
                else if (i == 32) {//nemocnica
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 33);
                }
                else if (i == 33) {//bytovka
                    insertOrdImageFromFile("./src/main/resources/images/obr" + Integer.toString(i) + ".gif", 47);
                }
            }
            catch (SQLException | IOException ex) {
                ex.printStackTrace();
            }
        }
    }


    /**
     * Delete record with OrdImage identified by id from database.
     * @param id id of picture in database
     * @throws SQLException
     */
    public void deleteOrdImageFromDb(int id) throws SQLException{
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();

        try(PreparedStatement preparedStatementDelete = connection.prepareStatement(SQL_DELETE)){
            preparedStatementDelete.setInt(1, id);
            preparedStatementDelete.executeUpdate();
            connection.commit();
        }
    }

    /**
     * Delete all records with OrdImages related to spatial entity from database.
     * @param spatialEntityId id of spatial entity in database
     * @throws SQLException
     */
    public void deleteOrdImagesOfSpatialEntityFromDb(int spatialEntityId) throws SQLException{
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();

        try(PreparedStatement preparedStatementDelete = connection.prepareStatement(SQL_DELETE_IMAGES)){
            preparedStatementDelete.setInt(1, spatialEntityId);
            preparedStatementDelete.executeUpdate();
            connection.commit();
        }
    }

    /**
     * Edit OrdImage according to parameters given in process string.
     * @param id id of the picture in database
     * @param process parameters determining operation on OrdImage, e.g. "rotate 90", "gamma 0.5 0.5 0.5"
     * @throws SQLException
     */
    public void processOrdImage(int id, String process) throws SQLException, IOException {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        final boolean currentAutoCommit = connection.getAutoCommit();
        connection.setAutoCommit(false);
        try {
            OrdImage imageProxy = getOrdImageForUpdate(id);
            imageProxy.process(process);

        } finally {
            //renew StillImage and information about it
            restoreStillImage(id);
            connection.commit();
            connection.setAutoCommit(currentAutoCommit);
        }

    }

    /**
     * Find 2 most similar title images of properties from database and id of spatial entity related to them.
     * @param propertyID id of spatial entity from database
     * @param weightAC weight of average color property
     * @param weightCH weight of color histogram property
     * @param weightPC weight of positional color property
     * @param weightTX weight of texture property
     * @return
     * @throws SQLException
     */
    public List<Integer> findSimilar(int propertyID, double weightAC, double weightCH, double weightPC, double weightTX) throws SQLException {
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        final List<Integer> similarProperties = new ArrayList<>();
        try (PreparedStatement preparedStatementFindSimilar = connection.prepareStatement(SQL_SIMILAR_IMAGE)) {
            preparedStatementFindSimilar.setDouble(1, weightAC);
            preparedStatementFindSimilar.setDouble(2, weightCH);
            preparedStatementFindSimilar.setDouble(3, weightPC);
            preparedStatementFindSimilar.setDouble(4, weightTX);
            preparedStatementFindSimilar.setInt(5, propertyID);
            try (ResultSet resultSet = preparedStatementFindSimilar.executeQuery()) {
                if (resultSet.next()) {
                    similarProperties.add((int)resultSet.getInt("id_spatial_entity"));
                    if (resultSet.next()) {
                        similarProperties.add((int)resultSet.getInt("id_spatial_entity"));
                    }
                }
            }
        }
       return similarProperties;
    }

    /**
     * Find id's and OrdImages of pictures related to property given by its id from database.
     * @param spatialEntityId id of the property from database
     * @return images hash map of OrdImages transformed to Images as values and their id's as keys
     * @throws SQLException
     */
        public HashMap<Integer, Image> findPicturesToSpatialEntity (int spatialEntityId) throws SQLException{
        DatabaseModel dbm = new DatabaseModel();
        Connection connection = dbm.getService().getConnection();
        final HashMap<Integer, Image> images = new HashMap<>();
        try (PreparedStatement preparedStatementFindImagesToProperty = connection.prepareStatement(SQL_SELECT_IMAGE_IMAGE_ID_TO_SPATIAL_ENTITY)) {
            preparedStatementFindImagesToProperty.setInt(1, spatialEntityId);
            try (ResultSet resultSet = preparedStatementFindImagesToProperty.executeQuery()) {
                while (resultSet.next()) {
                    final OracleResultSet oracleResultSet = (OracleResultSet) resultSet;
                    final OrdImage imgProxy = (OrdImage) oracleResultSet.getORAData(2, OrdImage.getORADataFactory());
                    final int id = (int) resultSet.getInt(1);
                    try {
                        BufferedImage bufferedImg = bufferedImg = ImageIO.read(new ByteArrayInputStream(imgProxy.getDataInByteArray()));
                        Image image = SwingFXUtils.toFXImage(bufferedImg, null);
                        images.put(id, image);
                    } catch (IOException e) {
                        return null; //??
                    }
                }
            }
        }
        return images;
    }

    public class NotExistException extends Exception{
    }

}