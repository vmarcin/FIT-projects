package upa.controller;

import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;

import javafx.event.ActionEvent;
import javafx.scene.image.Image;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Polygon;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import upa.model.DatabaseModel;
import upa.model.DatabaseService;
import upa.model.multimedia.Picture;

import java.io.IOException;
import java.sql.SQLException;
import java.util.concurrent.TimeUnit;

public class MainController {

    private DatabaseModel dbm;
    private Stage thisStage;
    private final RegistrationForm registrationForm;

    @FXML
    private BorderPane mainPane;
    @FXML
    private MenuItem closeItem;
    @FXML
    private MenuItem logoutItem;
    @FXML
    private ImageSlider imageSliderController;
    @FXML
    private MapPane mapPaneController;
    @FXML
    private InfoPane infoPaneController;
    @FXML
    private MenuItem saveButton;
    @FXML
    private MenuItem addRoad;
    @FXML
    private MenuItem addTramStop;
    @FXML
    private MenuItem addTreesCollection;
    @FXML
    private MenuItem delete;
    @FXML
    private MenuItem addShop;
    @FXML
    private MenuItem addSchool;
    @FXML
    private MenuItem addHospital;
    @FXML
    private MenuItem addBof;
    @FXML
    private MenuItem addPark;
    @FXML
    private MenuItem addEstate;
    @FXML
    private MenuItem addHouse;
    @FXML
    private MenuItem addFlat;
    @FXML
    private MenuItem initdb;

    public MainController(RegistrationForm registrationForm, DatabaseModel dbm) {
        this.registrationForm = registrationForm;
        this.dbm = dbm;

        thisStage = new Stage();

        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/Main.fxml"));
            loader.setController(this);
            thisStage.setScene(new Scene(loader.load()));
            thisStage.setTitle("Reality Office");
            thisStage.setResizable(false);
            thisStage.getIcons().add(new Image("/images/home-icon.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void showStage() {
        thisStage.show();
    }

    public void initialize() {
        imageSliderController.imageIndex = 0;
        imageSliderController.setMapPane(mapPaneController);
        imageSliderController.setInfoPane(infoPaneController);
        mapPaneController.setImageSlider(imageSliderController);
        mapPaneController.setBasicInfoPane(infoPaneController);
        infoPaneController.setImageSlider(imageSliderController);
        infoPaneController.setMapPane(mapPaneController);

        mapPaneController.setDbm(dbm);

        closeItem.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        logoutItem.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addRoad.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addTramStop.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addTreesCollection.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        delete.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });

        addShop.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addSchool.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addHospital.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addBof.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addPark.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addEstate.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addHouse.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
        addFlat.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });
    
        initdb.setOnAction(actionEvent -> {
            try { onButtonClicked(actionEvent); } catch (IOException e) {} });

        mapPaneController.initializeMap();
    }

    @FXML
    public void onButtonClicked(ActionEvent e) throws IOException {
        if(e.getSource().equals(closeItem)) {
            Stage stage = (Stage) mainPane.getScene().getWindow();
            stage.close();
        } else if(e.getSource().equals(logoutItem)) {
            Stage stage = (Stage) mainPane.getScene().getWindow();
            stage.setScene(new Scene(FXMLLoader.load(getClass().getResource("/fxml/RegistrationForm.fxml"))));
            stage.centerOnScreen();
        } else if(e.getSource().equals(addRoad)) {
            mapPaneController.setMode(2);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addTramStop)) {
            mapPaneController.setMode(3);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addTreesCollection)) {
            mapPaneController.setMode(4);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(delete)) {
            mapPaneController.deleteFocusedNode();
        } else if(e.getSource().equals(addShop)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(1);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addSchool)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(2);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addHospital)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(3);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addPark)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(4);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addBof)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(5);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addEstate)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(6);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(addHouse)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(7);
            mapPaneController.setFocusedOf();
        }else if(e.getSource().equals(addFlat)) {
            mapPaneController.setMode(1);
            mapPaneController.setBuildingType(8);
            mapPaneController.setFocusedOf();
        } else if(e.getSource().equals(initdb)) {
            DatabaseService dbs = dbm.getService();
            try {
                mapPaneController.deleteAllNodesFromMap();
                dbs.initFromSqlScript("src/main/resources/init.sql");
                dbs.initFromSqlScript("src/main/resources/initdata.sql");
                Picture picture = new Picture();
                picture.initPictures();
                dbs.getConnection().commit();
                mapPaneController.initializeMap();
            } catch (SQLException ex) {
                System.err.println("error init data");
            }
        }
    }
}
