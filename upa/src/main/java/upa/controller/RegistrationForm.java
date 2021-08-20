package upa.controller;

import com.sun.tools.javac.Main;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import upa.model.DatabaseModel;
import upa.model.DatabaseService;
import upa.model.multimedia.Picture;

import java.io.IOException;
import java.sql.SQLException;

public class RegistrationForm {
    private final Stage thisStage;

    @FXML
    private TextField nameField;

    @FXML
    private PasswordField passwordField;

    @FXML
    private TextField passwordFieldVisible;

    @FXML
    private Button submitButton;

    @FXML
    private CheckBox passToggle;

    public RegistrationForm() {
        thisStage = new Stage();
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/RegistrationForm.fxml"));
            loader.setController(this);

            thisStage.setScene(new Scene(loader.load()));

            thisStage.setTitle("Reality office");
            thisStage.getIcons().add(new Image("/images/home-icon.png"));
            thisStage.setResizable(false);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void showStage() {
        thisStage.show();
    }

    public void initialize() {
        this.toggleVisiblePassword(null);
        passToggle.setOnAction( event -> toggleVisiblePassword(event));
        submitButton.setOnAction(event -> {
            try {
                handleSubmitButtonAction(event);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        });
    }

    @FXML
    protected void handleSubmitButtonAction(ActionEvent event) throws IOException, SQLException {
        if(nameField.getText().isEmpty()) {
            nameField.setPromptText("Please enter a name!");
            return;
        }
//        passwordField.setText("xScUyJJR");
        if(passwordField.getText().isEmpty()) {
            passwordField.setPromptText("Please enter a password!");
            return;
        }

        DatabaseModel dbm = new DatabaseModel();
        DatabaseService dbs = dbm.getService();
        if (dbs.connectDatabase("gort.fit.vutbr.cz", "1521", "orclpdb", nameField.getText(), passwordField.getText())) {
            dbs.getConnection().setAutoCommit(false);
            dbs.initFromSqlScript("src/main/resources/init.sql");
            dbs.getConnection().commit();
            MainController mainController = new MainController(this, dbm);
            mainController.showStage();
            thisStage.close();
        } else {
            // TODO alert box
            nameField.setPromptText("Wrong name or password!");
        }
    }

    @FXML
    public void toggleVisiblePassword(ActionEvent e) {
        if(passToggle.isSelected()) {
            passwordFieldVisible.setText(passwordField.getText());
            passwordFieldVisible.setVisible(true);
            passwordField.setVisible(false);
            return;
        }
        passwordField.setText(passwordFieldVisible.getText());
        passwordField.setVisible(true);
        passwordFieldVisible.setVisible(false);
    }
}
