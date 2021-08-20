package upa;


import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import upa.controller.RegistrationForm;
import upa.model.DatabaseModel;

public class App extends Application {
    public DatabaseModel dbm = new DatabaseModel();

    public static void main(String[] args) throws Exception {

        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {

        RegistrationForm registrationForm = new RegistrationForm();
        registrationForm.showStage();
    }
}
