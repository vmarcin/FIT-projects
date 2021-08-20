package upa.controller;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.MenuButton;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TitledPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import upa.model.multimedia.Picture;

import javax.print.attribute.IntegerSyntax;
import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.*;


public class ImageSlider {

    @FXML
    private ImageView nextButton;
    @FXML
    private ImageView prevButton;
    @FXML
    private ImageView image;
    @FXML
    private Button addButton;
    @FXML
    private Button findButton;
    @FXML
    private MenuButton editButton;
    @FXML
    private MenuItem deleteItem;
    @FXML
    private MenuItem rotate;
    @FXML
    private MenuItem flip;
    @FXML
    private MenuItem mirror;
    @FXML
    private MenuItem gamma;
    @FXML
    private MenuItem contrast;
    private MapPane mapPane;
    private InfoPane infoPane;
    private final FileChooser fileChooser = new FileChooser();
    private List<Image> images;
    public int imageIndex;
    public List<Integer> imagesIds;

    public void initialize() {
        nextButton.setDisable(true);
        prevButton.setDisable(true);
        images = new ArrayList<>();
        imagesIds = new ArrayList<>();
    }

    public void setFindButtonDisable() {
        findButton.setDisable(true);
    }
    public void setFindButtonEnabled() {
        findButton.setDisable(false);
    }
    public void setInfoPane(InfoPane infoPane) { this.infoPane = infoPane; }
    public void setMapPane(MapPane mapPane) {this.mapPane = mapPane; }

    @FXML
    public void onArrowClicked(MouseEvent e) {
        if(e.getSource().equals(nextButton)) {
            if(imageIndex + 1 < images.size()) {
                prevButton.setImage(new Image((new File("src/main/resources/images/left.png")).toURI().toString()));
                prevButton.setDisable(false);
                image.setImage(images.get(++imageIndex));
                if(imageIndex == images.size() -1 ) {
                    nextButton.setImage(new Image((new File("src/main/resources/images/right-light.png")).toURI().toString()));
                    nextButton.setDisable(true);
                }
            } else {
                nextButton.setDisable(true);
            }
        } else if(e.getSource().equals(prevButton)) {
            if(imageIndex - 1 >= 0) {
                nextButton.setImage(new Image((new File("src/main/resources/images/right.png")).toURI().toString()));
                nextButton.setDisable(false);
                image.setImage(images.get(--imageIndex));
                if(imageIndex == 0) {
                    prevButton.setImage(new Image((new File("src/main/resources/images/left-light.png")).toURI().toString()));
                    prevButton.setDisable(true);
                }
            } else { prevButton.setDisable(true); }
        }
    }

    public File getImage(ActionEvent e) {
        Stage stage = (Stage) ((Button)e.getSource()).getScene().getWindow();
        return fileChooser.showOpenDialog(stage);
    }

    public void onButtonClicked(ActionEvent e) {
        Picture picture = new Picture();
        if(e.getSource().equals(addButton)) {
            File image = getImage(e);
            if (image != null){
                try {
                    picture.insertOrdImageFromFile(image.getAbsolutePath(), mapPane.getFocused());
                    loadImages(mapPane.getFocused(), 0);
                }
                catch (SQLException|IOException ex){
                    ex.printStackTrace();
                }
            }
        } else if (e.getSource().equals(findButton)) {
            //TODO - call with appropriate spatial entity id
            List<Integer> similarSpatialEntities = new ArrayList<>();
            try {
                similarSpatialEntities = picture.findSimilar(mapPane.getFocused(), 0.3, 0.3, 0.1, 0.3);
                if (similarSpatialEntities.size() >= 1) {
                    if(similarSpatialEntities.size() >=2) {
                    }
                }
            }
            catch (SQLException ex) {
                ex.printStackTrace();
            }

            Set<Integer> similarSpatialEntitiesSet = new HashSet<>(similarSpatialEntities);
            mapPane.setFocusedListOfEntities(similarSpatialEntitiesSet);
        } else if (e.getSource().equals(deleteItem)) {
            try  {
                picture.deleteOrdImageFromDb(imagesIds.get(imageIndex));
                loadImages(mapPane.getFocused(), 0);
            }
            catch (SQLException ex) {
                ex.printStackTrace();
            }
        } else if (e.getSource().equals(rotate)) {
            try {
                picture.processOrdImage(imagesIds.get(imageIndex), "rotate 90");
                loadImages(mapPane.getFocused(), imageIndex);
            }
            catch (SQLException|IOException ex) {
                ex.printStackTrace();
            }

        } else if (e.getSource().equals(flip)) {
            try {
                picture.processOrdImage(imagesIds.get(imageIndex), "flip");
                loadImages(mapPane.getFocused(), imageIndex);
            }
            catch(SQLException|IOException ex) {
                ex.printStackTrace();
            }

        } else if (e.getSource().equals(mirror)) {
            try {
                picture.processOrdImage(imagesIds.get(imageIndex), "mirror");
                loadImages(mapPane.getFocused(), imageIndex);
            }
            catch(SQLException|IOException ex) {
                ex.printStackTrace();
            }
        } else if (e.getSource().equals(gamma)) {
            infoPane.titledImageOperationsPane.setCollapsible(true);
            infoPane.accordionInfoPane.setExpandedPane(infoPane.titledImageOperationsPane);
            infoPane.titledImageOperationsPane.setCollapsible(false);
        } else if (e.getSource().equals(contrast)) {
            infoPane.titledImageOperationsPane.setCollapsible(true);
            infoPane.accordionInfoPane.setExpandedPane(infoPane.titledImageOperationsPane);
            infoPane.titledImageOperationsPane.setCollapsible(false);
        }
    }

    public void loadEmptyImage(){
        imageIndex = -1;
        if (!images.isEmpty()){
            images.clear();
        }
        if (!imagesIds.isEmpty()){
            imagesIds.clear();
        }
        image.setImage(null);
        prevButton.setImage(new Image((new File("src/main/resources/images/left-light.png")).toURI().toString()));
        prevButton.setDisable(true);
        nextButton.setImage(new Image((new File("src/main/resources/images/right-light.png")).toURI().toString()));
        nextButton.setDisable(true);

    }

    public void loadImages(int idOfBuilding, int indexToShow) {
        imageIndex = indexToShow;
        if (!images.isEmpty()){
            images.clear();
        }
        if (!imagesIds.isEmpty()){
            imagesIds.clear();
        }

        prevButton.setImage(new Image((new File("src/main/resources/images/left.png")).toURI().toString()));
        prevButton.setDisable(false);

        HashMap<Integer, Image> imagesHashMap = new HashMap<>();
        Picture picture = new Picture();
        try  {
            imagesHashMap = picture.findPicturesToSpatialEntity(idOfBuilding);
            Set<Integer> setOfIds = imagesHashMap.keySet();
            for(Integer id : setOfIds){
                imagesIds.add(id);
                images.add(imagesHashMap.get(id));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        if(images.isEmpty()) {
            findButton.setDisable(true);
            editButton.setDisable(true);
            image.setImage(null);
            prevButton.setImage(new Image((new File("src/main/resources/images/left-light.png")).toURI().toString()));
            prevButton.setDisable(true);
            nextButton.setImage(new Image((new File("src/main/resources/images/right-light.png")).toURI().toString()));
            nextButton.setDisable(true);
        }
        else{
            findButton.setDisable(false);
            editButton.setDisable(false);
            image.setImage(images.get(imageIndex));
            if(images.size() > 1) {
                nextButton.setImage(new Image((new File("src/main/resources/images/right.png")).toURI().toString()));
                nextButton.setDisable(false);
            }
            else {
                nextButton.setImage(new Image((new File("src/main/resources/images/right-light.png")).toURI().toString()));
                nextButton.setDisable(true);
                prevButton.setImage(new Image((new File("src/main/resources/images/left-light.png")).toURI().toString()));
                prevButton.setDisable(true);
            }
        }

        if (indexToShow == 0){
            prevButton.setImage(new Image((new File("src/main/resources/images/left-light.png")).toURI().toString()));
            prevButton.setDisable(true);
        }
        else if (indexToShow == images.size()-1) {
            nextButton.setImage(new Image((new File("src/main/resources/images/right-light.png")).toURI().toString()));
            nextButton.setDisable(true);
        }
    }
}
