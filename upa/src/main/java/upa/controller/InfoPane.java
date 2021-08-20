package upa.controller;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
//import javafx.scene.control.Accordion;
//import javafx.scene.control.Slider;
//import javafx.scene.control.TitledPane;

//import javafx.scene.layout.AnchorPane;
//import upa.model.multimedia.Picture;

//import java.io.IOException;
//import java.sql.SQLException;

import javafx.scene.control.*;
import javafx.scene.layout.AnchorPane;
import upa.model.multimedia.Picture;

import java.io.IOException;
import java.sql.SQLException;

public class InfoPane {
    private MapPane map;
    @FXML
    private TextArea bitelnumber;

    @FXML
    private TextArea biname;

    @FXML
    private TextArea biemail;

    @FXML
    private Label labelprice;

    @FXML
    private TextArea biaddress;

    @FXML
    private Label labelemail;

    @FXML
    private Label labeladdr;

    @FXML
    private TextArea bidesc;

    @FXML
    private Label labeloname;

    @FXML
    private TextArea bioname;

    @FXML
    private TextArea biprice;

    @FXML
    private Label labeltelnumber;

    @FXML
    private Label labeloi;

    @FXML
    private Label bitype;

    private ImageSlider imageSlider;

    @FXML
    private Button bisave;

    @FXML
    private Label areaLabel;

    @FXML
    private Label perimeterLabel;

    @FXML
    private Label schoolsResult;

    @FXML
    private Label shopsResult;

    @FXML
    private Label hospitalsResult;

    @FXML
    private Label parksResult;

    @FXML
    private Label tramStopsResult;

    @FXML
    private TextArea schoolsInput;

    @FXML
    private TextArea shopsInput;

    @FXML
    private TextArea hospitalsInput;

    @FXML
    private TextArea parksInput;

    @FXML
    private TextArea tramStopsInput;

    @FXML
    private CheckBox schoolsCheck;

    @FXML
    private CheckBox shopsCheck;

    @FXML
    private CheckBox hospitalsCheck;

    @FXML
    private CheckBox parksCheck;

    @FXML
    private CheckBox tramStopsCheck;

    @FXML
    private Button findButton;

      @FXML
    public Accordion accordionInfoPane;

    @FXML
    public TitledPane titledImageOperationsPane;

    @FXML
    public TitledPane titledBasicInfoPane;

    @FXML
    public TitledPane titledSpatialInfoPane;

    @FXML
    public TitledPane titledEntityModificationPane;

    @FXML
    public AnchorPane imageOperationsPane;

    @FXML
    public Slider gammaR;

    @FXML
    public Slider gammaG;

    @FXML
    public Slider gammaB;

    @FXML
    public Slider contrastR;

    @FXML
    public Slider contrastG;

    @FXML
    public Slider contrastB;

    public void setAreaLabel(String areaLabel) {
        this.areaLabel.setText(areaLabel);
    }

    public void setPerimeterLabel(String perimeterLabel) {
        this.perimeterLabel.setText(perimeterLabel);
    }

    public void setSchoolsResult(String schoolsResult) {
        this.schoolsResult.setText(schoolsResult);
    }

    public void setShopsResult(String shopsResult) {
        this.shopsResult.setText(shopsResult);
    }

    public void setHospitalsResult(String hospitalsResult) {
        this.hospitalsResult.setText(hospitalsResult);
    }

    public void setParksResult(String parksResult) {
        this.parksResult.setText(parksResult);
    }

    public void setTramStopsResult(String tramStopsResult) {
        this.tramStopsResult.setText(tramStopsResult);
    }

    public Integer getSchoolsInput() {
        return Integer.parseInt(schoolsInput.getText());
    }

    public Integer getShopsInput() {
        return Integer.parseInt(shopsInput.getText());
    }

    public Integer getHospitalsInput() {
        return Integer.parseInt(hospitalsInput.getText());
    }

    public Integer getParksInput() {
        return Integer.parseInt(parksInput.getText());
    }

    public Integer getTramStopsInput() {
        return Integer.parseInt(tramStopsInput.getText());
    }

    public boolean getSchoolsCheck() {
        return schoolsCheck.isSelected();
    }

    public boolean getShopsCheck() {
        return shopsCheck.isSelected();
    }

    public boolean getHospitalsCheck() {
        return hospitalsCheck.isSelected();
    }

    public boolean getParksCheck() {
        return parksCheck.isSelected();
    }

    public boolean getTramStopsCheck() {
        return tramStopsCheck.isSelected();
    }

    public void setBitelnumber(String bitelnumber) {
        this.bitelnumber.setText(bitelnumber);
    }

    public void setBiname(String biname) {
        this.biname.setText(biname);
    }

    public void setBiemail(String biemail) {
        this.biemail.setText(biemail);
    }

    public void setBiprice(String biprice) {
        this.biprice.setText(biprice);
    }

    public void setBitype(String bitype) {
        this.bitype.setText(bitype);
    }

    public void setBiaddress(String biaddress) {
        this.biaddress.setText(biaddress);
    }

    public void setBidesc(String bidesc) {
        this.bidesc.setText(bidesc);
    }

    public void setBioname(String bioname) {
        this.bioname.setText(bioname);
    }

    public String getBitelnumber() {
        return bitelnumber.getText();
    }

    public String getBiname() {
        return biname.getText();
    }

    public String getBiemail() { return biemail.getText(); }

    public String getBiprice() {
        return biprice.getText();
    }

    public String getBiaddress() {
        return biaddress.getText();
    }

    public String getBidesc() {
        return bidesc.getText();
    }

    public String getBioname() {
        return bioname.getText();
    }

    public void setEmtpy(){
        this.bitelnumber.setText("");
        this.biname.setText("");
        this.biemail.setText("");
        this.biprice.setText("");
        this.bitype.setText("");
        this.biaddress.setText("");
        this.bidesc.setText("");
        this.bioname.setText("");
    }

    public void setResults(double value, int index) {
        if (value == 1000.0) {
            value = 0.0;
        }
        switch (index) {
            case 0:
                setSchoolsResult(String.format("%.2f",value));
                break;
            case 1:
                setShopsResult(String.format("%.2f",value));
                break;
            case 2:
                setHospitalsResult(String.format("%.2f",value));
                break;
            case 3:
                setParksResult(String.format("%.2f",value));
                break;
            case 4:
                setTramStopsResult(String.format("%.2f",value));
                break;
        }
    }

    public void setResultsClear() {
        setSchoolsResult("-");
        setShopsResult("-");
        setHospitalsResult("-");
        setParksResult("-");
        setTramStopsResult("-");
    }

    public void onClick(ActionEvent e){
        if (e.getSource().equals(bisave)) {
            map.updateBasicInfoOfFocusedNode();
        }
        if(e.getSource().equals(findButton)){
            map.findNNearest();
        }
    }

    public void setMapPane(MapPane map) {
        this.map = map;
    }

    public void hidePropertyInfo(){
        biaddress.setVisible(false);
        biprice.setVisible(false);
        bioname.setVisible(false);
        biemail.setVisible(false);
        bitelnumber.setVisible(false);
        labeladdr.setVisible(false);
        labelprice.setVisible(false);
        labeloi.setVisible(false);
        labeloname.setVisible(false);
        labelemail.setVisible(false);
        labeltelnumber.setVisible(false);
    }

    public void showPropertyInfo(){
        biaddress.setVisible(true);
        biprice.setVisible(true);
        bioname.setVisible(true);
        biemail.setVisible(true);
        bitelnumber.setVisible(true);
        labeladdr.setVisible(true);
        labelprice.setVisible(true);
        labeloi.setVisible(true);
        labeloname.setVisible(true);
        labelemail.setVisible(true);
        labeltelnumber.setVisible(true);
    }
    public void initialize(){
        titledImageOperationsPane.setCollapsible(false);
    }

    public void setImageSlider(ImageSlider imageSlider) { this.imageSlider = imageSlider; }//

    /**
     * Compute coefficients needed for change of gamma and call function to perform change with coefficients on the image in database.
     * @param event
     * @throws SQLException
     * @throws IOException
     */
    @FXML
    public void gammaApplyClick(ActionEvent event) {

        final double red = ((double)this.gammaR.getValue()) / 100.0;
        final double green = ((double)this.gammaG.getValue()) / 100.0;
        final double blue = ((double)this.gammaB.getValue()) / 100.0;
        Picture picture = new Picture();
        try {
            picture.processOrdImage(imageSlider.imagesIds.get(imageSlider.imageIndex), "gamma " + String.valueOf(red) + " " + String.valueOf(green) + " " + String.valueOf(blue));
        }
        catch (SQLException | IOException ex){
            titledImageOperationsPane.setCollapsible(true);
            accordionInfoPane.setExpandedPane(null);
            titledImageOperationsPane.setCollapsible(false);
        }
        finally {
            titledImageOperationsPane.setCollapsible(true);
            accordionInfoPane.setExpandedPane(null);
            titledImageOperationsPane.setCollapsible(false);
            imageSlider.loadImages(map.getFocused(), imageSlider.imageIndex);
        }
    }

    /**
     * Finds coefficients needed for change of contrast and call function to perform change with coefficients on the image in database.
     * @param event
     * @throws SQLException
     * @throws IOException
     */
    @FXML
    public void contrastApplyClick(ActionEvent event)
    {
        final double red = ((double)this.contrastR.getValue()) / 2.0;
        final double green = ((double)this.contrastG.getValue()) / 2.0;
        final double blue = ((double)this.contrastB.getValue()) / 2.0;
        Picture picture = new Picture();
        try {
            picture.processOrdImage(imageSlider.imagesIds.get(imageSlider.imageIndex), "contrast " + String.valueOf(red) + " " + String.valueOf(green) + " " + String.valueOf(blue));
        }
        catch (SQLException|IOException e){
            titledImageOperationsPane.setCollapsible(true);
            accordionInfoPane.setExpandedPane(null);
            titledImageOperationsPane.setCollapsible(false);
        }
        finally {
            titledImageOperationsPane.setCollapsible(true);
            accordionInfoPane.setExpandedPane(null);
            titledImageOperationsPane.setCollapsible(false);
            imageSlider.loadImages(map.getFocused(), imageSlider.imageIndex);
        }
    }

    @FXML
    public void cancelButtonClick(ActionEvent event)
    {
        titledImageOperationsPane.setCollapsible(true);
        accordionInfoPane.setExpandedPane(null);
        titledImageOperationsPane.setCollapsible(false);
    }

}
