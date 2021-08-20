package upa.controller;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.stage.Stage;

import java.io.IOException;

public class InputDialogProperty {
    private final Stage thisStage;

    @FXML
    private TextField name;
    @FXML
    private TextField desc;
    @FXML
    private TextField address;
    @FXML
    private TextField price;
    @FXML
    private Button save;

    private MapPane mapPane;


    public InputDialogProperty(MapPane m) {
        this.mapPane = m;
        thisStage = new Stage();

        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/InputDialogProperty.fxml"));
            loader.setController(this);

            thisStage.setScene(new Scene(loader.load()));

            thisStage.setTitle("Property Info");
            thisStage.getIcons().add(new Image("/images/home-icon.png"));
            thisStage.setResizable(false);
        } catch (IOException e){}
    }

    public void showStage() {thisStage.show();}

    public void closeStage() {thisStage.close();}

    public void initialize() {
        save.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {
                handleSaveButtonAction(actionEvent);
            }
        });
    }

    public void handleSaveButtonAction(ActionEvent event) {
        String name = this.name.getText();
        String desc = this.desc.getText();
        String address = this.address.getText();
        String price = this.price.getText();
        int id = mapPane.getMaxId();

        mapPane.setPropertyInputData(name, desc, address, price, this, id);
    }
}
