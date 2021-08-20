package upa.controller;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.ObservableList;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.control.Alert;
import javafx.scene.effect.DropShadow;
import javafx.scene.input.*;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import javafx.util.Pair;
import upa.model.DatabaseModel;
import upa.model.SpatialEntityModel;
import upa.model.multimedia.Picture;

import java.sql.SQLException;
import java.util.*;

public class MapPane {
    final private SpatialEntityModel sem = new SpatialEntityModel();
    private DatabaseModel dbm;
    private InputDialogProperty inputDialogProperty;
    private InputDialogSpatialEntity inputDialogSpatialEntity;
    private InfoPane basicInfoPane;

    @FXML
    private AnchorPane map;
    // fields used for interactive adding of polygons and polylines
    private Polygon polygon;
    private Polyline polyline;
    // map object ID
    private int maxId = 1;
    // instance of image slider pane used for loading images
    private ImageSlider imageSlider;
    // map mode (eg. mode=1 add polygon, etc.)
    private int mode = 0;
    // type of building
    private int buildingType = 0;
    // stores a click position in the pane (map)
    final ObjectProperty<Point2D> mousePosition = new SimpleObjectProperty<>();
    // list of control points of polygon/polyline
    private ArrayList<Circle> cList = new ArrayList<>();
    // list of circles
    private ArrayList<Circle> trees = new ArrayList<>();
    // stores ID of actually focused object in the map
    private int focused = -1;
    // table to store all map nodes
    private Hashtable<Integer, Node> mapNodes = new Hashtable<Integer, Node>();

    private boolean wasDraged = false;
    private Double oldLayoutX = 0.0;
    private Double oldLayoutY = 0.0;
    private Double oldLayoutCX = 0.0;
    private Double oldLayoutCY = 0.0;

    public int getFocused() {
        return this.focused;
    }

    public void deleteAllNodesFromMap() {
        int size  = map.getChildren().size();
        for (int i = 0; i < size ; i++) {
            map.getChildren().remove(map.getChildren().size()-1);
        }
        mapNodes.clear();
        maxId = 1;
        clearControlPoints();
        focused = -1;
    }

    public void setFocusedListOfEntities(Set<Integer> nodes) {
        DropShadow dp = new DropShadow(10.0f, 0.0f, 0.0f, Color.RED);
        Set<Integer> keys = mapNodes.keySet();

        for(Integer k: keys)
            mapNodes.get(k).setEffect(null);
        focused = -1;
        clearControlPoints();

        for(Integer n: nodes)
            mapNodes.get(n).setEffect(dp);
    }


    public void findNNearest() {
        HashMap<Integer,Double> distances;
        ArrayList<Boolean> checks = new ArrayList<>();
        ArrayList<String> types = new ArrayList<>();
        ArrayList<Integer> n = new ArrayList<>();
        Set<Integer> nearestObjects = new HashSet<>();


        types.add("school");
        types.add("shop");
        types.add("hospital");
        types.add("park");
        types.add("tram stop");

        checks.add(basicInfoPane.getSchoolsCheck());
        checks.add(basicInfoPane.getShopsCheck());
        checks.add(basicInfoPane.getHospitalsCheck());
        checks.add(basicInfoPane.getParksCheck());
        checks.add(basicInfoPane.getTramStopsCheck());

        n.add(basicInfoPane.getSchoolsInput());
        n.add(basicInfoPane.getShopsInput());
        n.add(basicInfoPane.getHospitalsInput());
        n.add(basicInfoPane.getParksInput());
        n.add(basicInfoPane.getTramStopsInput());


        if(focused != -1) {
                for (int i = 0; i < types.size(); i++) {
                    if (checks.get(i)) {
                        distances = sem.getNNearestNeighboursByType(focused, n.get(i), types.get(i));
                        List<Double> d = new ArrayList<>(distances.values());
                        Collections.sort(d);
                        if(!d.isEmpty()) {
                            basicInfoPane.setResults(d.get(0) , i);
                            nearestObjects.addAll(distances.keySet());
                        }
                    }
                }
                setFocusedListOfEntities(nearestObjects);
        }
    }


    public void updateBasicInfoOfFocusedNode(){
        String name = basicInfoPane.getBiname();
        String description = basicInfoPane.getBidesc();

        try {
            if (sem.getType(focused).equals("property")){
                String oname = "";
                String osurname = "";
                if(!basicInfoPane.getBioname().equals("")) {
                    oname = basicInfoPane.getBioname().split(" ")[0];
                    osurname = basicInfoPane.getBioname().split(" ")[1];
                }
                String telnum = basicInfoPane.getBitelnumber();
                String email = basicInfoPane.getBiemail();
                String address = basicInfoPane.getBiaddress();
                String price = basicInfoPane.getBiprice();

                HashMap<String, String> propertyInfo = sem.getPropertyInfo(focused);
                sem.updateInfoSpatialEntity(focused, Map.of("name", name, "description", description));
                sem.updateInfoProperty(focused, Map.of("address", address, "price", price));
                sem.om.updateOwner(Integer.parseInt(propertyInfo.get("id_owner")), oname, osurname, email, telnum);
            } else {
                sem.updateInfoSpatialEntity(focused, Map.of("name", name, "description", description));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void fillBasicInfoOfFocusedNode(int id){
        try {
            basicInfoPane.setEmtpy();
            if(sem.getType(id).equals("property")){
                HashMap<String, String> propertyInfo = sem.getPropertyInfo(id);
                basicInfoPane.setAreaLabel(propertyInfo.get("area"));
                basicInfoPane.setPerimeterLabel(propertyInfo.get("length"));

                basicInfoPane.showPropertyInfo();
                basicInfoPane.setBiname(propertyInfo.get("name"));
                basicInfoPane.setBidesc(propertyInfo.get("description"));
                basicInfoPane.setBitype("property (" + propertyInfo.get("property_type") + ")");
                basicInfoPane.setBiaddress(propertyInfo.get("address"));
                basicInfoPane.setBiprice(propertyInfo.get("price"));
                try {
                    HashMap<String, String> ownerInfo = sem.om.getOwner(Integer.parseInt(propertyInfo.get("id_owner")));
                    basicInfoPane.setBioname(ownerInfo.get("name") + " " + ownerInfo.get("surname"));
                    basicInfoPane.setBiemail(ownerInfo.get("email"));
                    basicInfoPane.setBitelnumber(ownerInfo.get("telnum"));
                }catch (Exception e){}
            } else {
                basicInfoPane.setAreaLabel("-");
                basicInfoPane.setPerimeterLabel("-");
                basicInfoPane.hidePropertyInfo();
                HashMap<String, String> spatialEntityInfo = sem.getSpatialEntityInfo(id);
                basicInfoPane.setBiname(spatialEntityInfo.get("name"));
                basicInfoPane.setBidesc(spatialEntityInfo.get("description"));
                basicInfoPane.setBitype(spatialEntityInfo.get("type"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void setBasicInfoPane(InfoPane basicInfoPane){
        this.basicInfoPane = basicInfoPane;
    }

    public int getMaxId(){
        return maxId - 1;
    }

    public void setPropertyInputData(String name, String desc, String address, String price, InputDialogProperty i, int id) {
        Map.of("name", name, "description", desc, "address", address, "price", price);
        try {
            sem.updateInfoSpatialEntity(id, Map.of("name", name, "description", desc));
            sem.updateInfoProperty(id, Map.of("address", address, "price", price));
        } catch (SQLException e) { }

        i.closeStage();
    }

    public void setSpatialEntityInputData(String name, String desc, InputDialogSpatialEntity i, int id) {
        Map.of("name", name, "description", desc);
        try {
            sem.updateInfoSpatialEntity(id, Map.of("name", name, "description", desc));
        } catch (SQLException e) { }

        i.closeStage();
    }


    public void setPolygonHandlers(Polygon p) {
        p.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                oldLayoutX = p.getLayoutX();
                oldLayoutY = p.getLayoutY();
                setFocusedOnClick(mouseEvent);
                ((Polygon) mouseEvent.getSource()).toFront();
                clearControlPoints();
                setFocusedOnClick(mouseEvent);
                mousePosition.set(new Point2D(mouseEvent.getSceneX(), mouseEvent.getSceneY()));
            }
        });
        p.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                if(mouseEvent.getButton() == MouseButton.PRIMARY)
                    onDragNode(mouseEvent, false);
            }
        });
        p.setOnDragDetected(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                wasDraged = true;
            }
        });
        p.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                if(wasDraged) {
                    wasDraged = false;
                    Polygon normalizedPolygon = updatePolygonCoordinates(p.getPoints(), p);
                    if (!sem.updateJGeometry(focused, normalizedPolygon, sem.getType(focused))) {
                        p.setLayoutX(oldLayoutX);
                        p.setLayoutY(oldLayoutY);
                    }
                }
            }
        });
    }

    public void setBlockLayer() {
        for (Integer i: sem.getAllBlockOfFlats()) {
            Polygon block = (Polygon)mapNodes.get(i);
            if(sem.isBlockEmpty(i)){
                block.toFront();
            } else {
                block.toBack();
            }
        }
    }

    public void setMultipointHandlers(Group group) {
        group.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                oldLayoutX = group.getLayoutX();
                oldLayoutY = group.getLayoutY();
                if(mouseEvent.getButton() == MouseButton.PRIMARY) {
                    ((Group) mouseEvent.getSource()).toFront();
                    setFocusedOnClick(mouseEvent);
                    clearControlPoints();
                    mousePosition.set(new Point2D(mouseEvent.getSceneX(), mouseEvent.getSceneY()));
                }
            }
        });
        group.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                onDragNode(mouseEvent,false);
            }
        });
        group.setOnDragDetected(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                wasDraged = true;
            }
        });
        group.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                if(wasDraged) {
                    wasDraged = false;
                    if (!sem.updateJGeometry(focused, groupToArray(group, group.getLayoutX(), group.getLayoutY()), sem.getType(focused))) {
                        group.setLayoutX(oldLayoutX);
                        group.setLayoutY(oldLayoutY);
                    }
                }
            }
        });
    }

    public void setTreeHandler(Circle c) {
        c.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                Group g = (Group)((Circle)mouseEvent.getSource()).getParent();
                setFocusedOnClick(g);
                if(mouseEvent.getButton() == MouseButton.MIDDLE) {
                    if(mode != 4) {
                        map.getChildren().remove(mouseEvent.getSource());
                        g.getChildren().remove(mouseEvent.getSource());
                        if(g.getChildren().size() == 0) {
                            sem.deleteSpatialEntity(focused);
                            mapNodes.remove(focused);
                            focused=-1;
                        } else {
                            sem.updateJGeometry(focused, groupToArray(g, g.getLayoutX(), g.getLayoutY()), sem.getType(focused));
                        }
                    }
                }
            }
        });
    }

    public void setTramLinesHandlers(Polyline p) {
        p.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                oldLayoutX = p.getLayoutX();
                oldLayoutY = p.getLayoutY();
                ((Polyline) mouseEvent.getSource()).toFront();
                clearControlPoints();
                setFocusedOnClick(mouseEvent);
                mousePosition.set(new Point2D(mouseEvent.getSceneX(), mouseEvent.getSceneY()));
            }
        });
        p.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                onDragNode(mouseEvent, false);
            }
        });
        p.setOnDragDetected(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                wasDraged = true;
            }
        });
        p.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                if(wasDraged) {
                    wasDraged = false;
                    Polyline polyline = updatePolylineCoordinates(p.getPoints(), p);
                    if (!sem.updateJGeometry(focused, polyline, sem.getType(focused))) {
                        p.setLayoutX(oldLayoutX);
                        p.setLayoutY(oldLayoutY);
                    }
                }
            }
        });
    }

    public ArrayList<Double> addLayoutOffset(List<Double> points, Node p) {
        ArrayList<Double> updatedPoints = new ArrayList<>();
        for (int i = 0; i < (points.size()); i+=2) {
            updatedPoints.add(points.get(i) + p.getLayoutX());
            updatedPoints.add(points.get(i+1) + p.getLayoutY());
        }
        return updatedPoints;
    }

    public void joinEstates(List<Integer> oldIds, int newId) {
        ArrayList<Double> points;
        Polygon estate = new Polygon();

        setFocusedOf();

        for(Integer id: oldIds) {
            map.getChildren().remove(mapNodes.get(id));
            mapNodes.remove(id);
        }
        try {
            points = normalizeCoordinatesDB(((Polygon)sem.getJavaFxObject(newId)).getPoints());
            int index = points.size();
            points.remove(index-1);
            points.remove(index-2);
            estate.getPoints().addAll(points);
            estate.setFill(Color.SANDYBROWN);
            estate.setStroke(Color.GRAY);
            estate.setStrokeWidth(2);
            estate.setStrokeLineCap(StrokeLineCap.ROUND);
            setPolygonHandlers(estate);
            map.getChildren().add(estate);
            mapNodes.put(newId, estate);
        } catch (SQLException e) {}

    }

    public void createBuildings(List<Integer> listOfIds, Color color) {
        ArrayList<Double> points;
        for (Integer id: listOfIds) {
            try {
                Polygon p = new Polygon();
                points = normalizeCoordinatesDB(((Polygon)sem.getJavaFxObject(id)).getPoints());
                int index = points.size();
                points.remove(index-1);
                points.remove(index-2);
                p.getPoints().addAll(points);
                p.setFill(color);
                p.setStroke(Color.GRAY);
                p.setStrokeWidth(2);
                p.setStrokeLineCap(StrokeLineCap.ROUND);
                if(!color.equals(Color.rgb(200, 200, 200, 0.2))) {
                    setPolygonHandlers(p);
                }
                map.getChildren().add(p);
                // store building into hash map
                mapNodes.put(id, p);

            } catch (SQLException e) { }
        }
    }

    public void createTramLines(List<Integer> listofIds) {
        ArrayList<Double> points;
        for (Integer id: listofIds) {
            try {
                Polyline p = new Polyline();
                points = normalizeCoordinatesDB(((Polyline)sem.getJavaFxObject(id)).getPoints());
                p.getPoints().addAll(points);
                p.setStroke(Color.LIGHTCORAL);
                p.setStrokeWidth(3);
                p.setStrokeLineCap(StrokeLineCap.ROUND);

                setTramLinesHandlers(p);
                // tram line added to map
                map.getChildren().add(p);
                // reference to tram line stored into hash map
                mapNodes.put(id, p);
            } catch (SQLException e) {}
        }
    }

    public void createTramStops(List<Integer> listofIds) {
        for (Integer id: listofIds) {
            try {
                Point2D tramStop;
                tramStop = (Point2D) sem.getJavaFxObject(id);
                createTramStop(tramStop.getX(), (600 - tramStop.getY()), id);
            } catch (SQLException e) { }
        }
    }

    public void createTreesCollections(List<Integer> listofIds) {
        ArrayList<Point2D> trees = new ArrayList<>();
        for (Integer id: listofIds) {
            try {
                Group group = new Group();
                trees.addAll((List<Point2D>)sem.getJavaFxObject(id));
                for (Point2D point: trees) {
                    Circle c = new Circle(point.getX(), (600 - point.getY()), 8.0);
                    c.setFill(Color.LIGHTGREEN);
                    c.setStroke(Color.DARKGREEN);
                    c.setStrokeWidth(3.0);
                    setTreeHandler(c);
                    group.getChildren().add(c);
                }
                setMultipointHandlers(group);
                trees.clear();
                mapNodes.put(id, group);
                map.getChildren().add(group);
            } catch (SQLException e) { }
        }
    }

    public void initializeMap(){
        // create all polygons
        createBuildings(sem.getAllEstatesAsSpatialEntities(), Color.rgb(200, 200, 200, 0.2));
        createBuildings(sem.getAllShops(), Color.LIGHTBLUE);
        createBuildings(sem.getAllSchools(), Color.GOLD);
        createBuildings(sem.getAllHospitals(), Color.LIGHTSALMON);
        createBuildings(sem.getAllParks(), Color.DARKSEAGREEN);
        createBuildings(sem.getAllBlockOfFlats(), Color.ROSYBROWN);
        createBuildings(sem.getAllEstates(), Color.SANDYBROWN);
        createBuildings(sem.getAllHouses(), Color.TAN);
//        // create all polylines
        createTramLines(sem.getAllTramLines());
//        // create all tramstops
        createTramStops(sem.getAllTramStops());
//        // create all trees collections
        createTreesCollections(sem.getAllTrees());

        try {
            maxId = dbm.getNewId("SpatialEntities");
        } catch (SQLException e) { }
    }


    public ArrayList<Double> normalizeCoordinatesDB(List<Double> list) {
        ArrayList<Double> newList = new ArrayList<>();
        newList.addAll(list);
        double oldValue;
        for (int i = 1; i < list.size(); i+=2) {
            oldValue = list.get(i);
            newList.set(i, (600 - oldValue));
        }
        return newList;
    }

    public ArrayList<Double> reversePoints(List<Double> points) {
        ArrayList<Double> reversedPoints = new ArrayList<>(Arrays.asList(new Double[points.size()]));
        for (int i = points.size()-1,j=0; i > 0 ; i-=2,j+=2) {
            reversedPoints.set(j+1,points.get(i));
            reversedPoints.set(j, points.get(i-1));
        }
        return reversedPoints;
    }

    public Polygon normalizePolygonDB(Polygon polygon) {
        Polygon normalizedPolygon = new Polygon();
        normalizedPolygon.getPoints().addAll(normalizeCoordinatesDB(polygon.getPoints()));
        normalizedPolygon.getPoints().add(polygon.getPoints().get(0));
        normalizedPolygon.getPoints().add(600 - polygon.getPoints().get(1));
        return normalizedPolygon;
    }

    public void setImageSlider(ImageSlider imageSlider) { this.imageSlider = imageSlider; }

    public void setMode(int mode) { this.mode = mode; }

    public void setDbm(DatabaseModel dbm) { this.dbm = dbm; }

    public void setBuildingType(int buildingType) {this.buildingType = buildingType;}

    public void setFocusedOnClick(MouseEvent m) {
        Node p = ((Node) m.getSource());
        DropShadow dp = new DropShadow(2.0f, 5.0f, 5.0f, Color.DARKGREY);
        Set<Integer> keys = mapNodes.keySet();
        for (Integer k: keys) {
            if(mapNodes.get(k).equals(p)){
                if(p instanceof Polygon) {
                    imageSlider.loadImages(focused, 0);
                }
                else {
                    imageSlider.loadEmptyImage();
                }
                if(sem.getType(k).equals("property")) {
                    imageSlider.setFindButtonEnabled();
                } else  {
                    imageSlider.setFindButtonDisable();
                }
                p.setEffect(dp);
                focused = k;
                fillBasicInfoOfFocusedNode(k);
                basicInfoPane.setResultsClear();
            } else { mapNodes.get(k).setEffect(null); }
        }
    }

    public void setFocusedOnClick(Node n) {
        DropShadow dp = new DropShadow(2.0f, 5.0f, 5.0f, Color.DARKGREY);
        Set<Integer> keys = mapNodes.keySet();
        for (Integer k: keys) {
            if(mapNodes.get(k).equals(n)) {
                n.setEffect(dp);
                if(sem.getType(k).equals("property")) {
                    imageSlider.setFindButtonEnabled();
                } else  {
                    imageSlider.setFindButtonDisable();
                }
                focused = k;
                fillBasicInfoOfFocusedNode(k);
                basicInfoPane.setResultsClear();
            } else { mapNodes.get(k).setEffect(null); }
        }
    }

    public void setFocusedOf() {
        if(focused != -1) {
            mapNodes.get(focused).setEffect(null);
            focused = -1;
            clearControlPoints();
        }
    }

    @FXML
    public void onMouseMoved(MouseEvent e) {
        if (mode == 1) {
            if (polygon != null) {
                polygon.toFront();
                polygon.getPoints().set(polygon.getPoints().size() - 2, e.getX());
                polygon.getPoints().set(polygon.getPoints().size() - 1, e.getY());
            }
        } else if(mode == 2) {
            if(polyline != null) {
                polyline.getPoints().set(polyline.getPoints().size()-2, e.getX());
                polyline.getPoints().set(polyline.getPoints().size()-1, e.getY());
            }
        }
    }

    public void onDragNode(MouseEvent m, Boolean controlPoint) {
        double deltaX = m.getSceneX() - mousePosition.get().getX();
        double deltaY = m.getSceneY() - mousePosition.get().getY();

        Node movedNode = ((Node) m.getSource());

        double minX = movedNode.boundsInParentProperty().getValue().getMinX();
        double maxX = movedNode.boundsInParentProperty().getValue().getMaxX();
        double minY = movedNode.boundsInParentProperty().getValue().getMinY();
        double maxY = movedNode.boundsInParentProperty().getValue().getMaxY();

        double newX = movedNode.getLayoutX() + deltaX;
        double newY = movedNode.getLayoutY() + deltaY;

        if (((minX + deltaX) > 0.0 && (maxX + deltaX) < map.getWidth()) &&
                ((minY + deltaY) > 0.0 && (maxY + deltaY) < map.getHeight())) {
            movedNode.setLayoutX(newX);
            movedNode.setLayoutY(newY);
            // move of control point which is tied up with polygon/polyline point
            if (controlPoint) {
                if(mapNodes.get(focused) instanceof  Polygon) {
                    Polygon polygon = (Polygon) mapNodes.get(focused);
                    Circle movedCircle = (Circle) movedNode;
                    int index = cList.indexOf(m.getSource()) * 2;
                    polygon.getPoints().set(index, (movedCircle.getCenterX() - polygon.getLayoutX()) + movedCircle.getLayoutX());
                    polygon.getPoints().set(index + 1, (movedCircle.getCenterY() - polygon.getLayoutY()) + movedCircle.getLayoutY());
                } else if (mapNodes.get(focused) instanceof Polyline) {
                    Polyline polyline = (Polyline) mapNodes.get(focused);
                    Circle movedCircle = (Circle) movedNode;
                    int index = cList.indexOf(m.getSource()) * 2;
                    polyline.getPoints().set(index, (movedCircle.getCenterX() - polyline.getLayoutX()) + movedCircle.getLayoutX());
                    polyline.getPoints().set(index + 1, (movedCircle.getCenterY() - polyline.getLayoutY()) + movedCircle.getLayoutY());
                }
            }
        }
        mousePosition.set(new Point2D(m.getSceneX(), m.getSceneY()));
    }

    public Pair<Boolean,Color> insertBuildingByType(Polygon p) {
        switch (buildingType) {
            case 1: return new Pair<>(sem.insertShop(p), Color.LIGHTBLUE);
            case 2: return new Pair<>(sem.insertSchool(p),Color.GOLD);
            case 3: return new Pair<>(sem.insertHospital(p),Color.LIGHTSALMON);
            case 4: return new Pair<>(sem.insertPark(p),Color.DARKSEAGREEN);
            case 5: return new Pair<>(sem.insertBlockOfFlats(p),Color.ROSYBROWN);
            case 6: return new Pair<>(sem.insertEstate(p, this),Color.SANDYBROWN);
            case 7: return new Pair<>(sem.insertHouse(p),Color.TAN);
            case 8: return new Pair<>(sem.insertFlat(p),Color.KHAKI);
        }
        return null;
    }


    public void createBuilding(MouseEvent e) {
        boolean insertionFailed = false;
        if (e.getButton() == MouseButton.PRIMARY) {
            if (polygon == null) {
                polygon = new Polygon();
                polygon.setStroke(Color.GRAY);
                polygon.setStrokeWidth(3);
                polygon.setStrokeLineCap(StrokeLineCap.ROUND);
                polygon.setFill(Color.LIGHTGRAY);
                polygon.getPoints().addAll(e.getX(), e.getY());
                map.getChildren().add(polygon);
            }
            polygon.getPoints().addAll(e.getX(), e.getY());
        } else {
            if (polygon != null) {
                Polygon p = new Polygon();
                p.setStroke(Color.GRAY);
                p.setStrokeWidth(2);
                p.setStrokeLineCap(StrokeLineCap.ROUND);
                p.getPoints().addAll(polygon.getPoints().subList(0, polygon.getPoints().size() -2));
                map.getChildren().remove(polygon);
                polygon = null;
                if(p.getPoints().size() > 4) {
                    setPolygonHandlers(p);
                    Pair<Boolean, Color> pair = insertBuildingByType(normalizePolygonDB(p));
                    p.setFill(pair.getValue());
                    if (pair.getKey()) {
                        try {
                            if (sem.checkJGeometryValidity()) {
                                map.getChildren().add(p);
                                // store building into hash map
                                mapNodes.put(maxId++, p);
                                if (buildingType == 6){
                                    sem.joinEstatesBackend((maxId-1), this);
                                }
                            } else {
                                int oldId = dbm.getNewId("SpatialEntities") - 1;
                                if (sem.getType(oldId).equals("property"))
                                    sem.deleteProperty(oldId);
                                else
                                    sem.deleteSpatialEntity(oldId);
                                Polygon normPolygon = normalizePolygonDB(p);
                                Polygon revePolygon = new Polygon();
                                revePolygon.getPoints().addAll(reversePoints(normPolygon.getPoints()));
                                if (insertBuildingByType(revePolygon).getKey()) {
                                    p.getPoints().clear();
                                    ArrayList<Double> points = normalizeCoordinatesDB(revePolygon.getPoints());
                                    p.getPoints().addAll(points.subList(0, points.size() - 2));
                                    if (sem.checkJGeometryValidity()) {
                                        map.getChildren().add(p);
                                        // store building into hash map
                                        mapNodes.put(maxId++, p);
                                        if (buildingType == 6){
                                            sem.joinEstatesBackend((maxId-1), this);
                                        }
                                    } else {
                                        oldId = dbm.getNewId("SpatialEntities") - 1;
                                        if (sem.getType(oldId).equals("property"))
                                            sem.deleteProperty(oldId);
                                        else
                                            sem.deleteSpatialEntity(oldId);
                                        insertionFailed = true;
                                    }
                                } else {
                                    insertionFailed = true;
                                }
                            }
                        } catch (SQLException ex) {
                        }
                    } else {
                        insertionFailed = true;
                    }
                    // exit insert mode
                    if (!insertionFailed) {
                        if (sem.getType(getMaxId()).equals("property")) {
                            inputDialogProperty = new InputDialogProperty(this);
                            inputDialogProperty.showStage();
                            setFocusedOf();
                        } else {
                            inputDialogSpatialEntity = new InputDialogSpatialEntity(this);
                            inputDialogSpatialEntity.showStage();
                            setFocusedOf();
                        }
                    }
                    setBlockLayer();
                }
                this.mode = 0;
            }
        }
    }

    public void createTramStop(double ex, double ey, int id) {
        double x = ex;
        double y = ey;

        Circle c = new Circle(x, y, 7.0);
        c.setFill(Color.RED);
        c.setStroke(Color.BLACK);
        c.setStrokeWidth(2.0);
        c.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                oldLayoutCX = c.getLayoutX();
                oldLayoutCY = c.getLayoutY();
                ((Circle) mouseEvent.getSource()).toFront();
                setFocusedOnClick(mouseEvent);
                clearControlPoints();
                mousePosition.set(new Point2D(mouseEvent.getSceneX(), mouseEvent.getSceneY()));
            }
        });
        c.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                onDragNode(mouseEvent, false);
            }
        });
        c.setOnDragDetected(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                wasDraged = true;
            }
        });
        c.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                if(wasDraged) {
                    wasDraged = false;
                    Point2D tramStop = new Point2D(c.getCenterX() + c.getLayoutX(), 600 - (c.getCenterY() + c.getLayoutY()));
                    if (!sem.updateJGeometry(focused, tramStop, sem.getType(focused))) {
                        c.setLayoutX(oldLayoutCX);
                        c.setLayoutY(oldLayoutCY);
                    }
                }
            }
        });
        if(id == -1) {
            if(sem.insertTramStop(new Point2D(x,(600 - y)))) {
                map.getChildren().add(c);
                mapNodes.put(maxId, c);
                inputDialogSpatialEntity = new InputDialogSpatialEntity(this);
                inputDialogSpatialEntity.showStage();
                setFocusedOf();
            }
        } else {
            map.getChildren().addAll(c);
            mapNodes.put(id, c);
        }
        this.mode = 0;
    }

    public void createTramLine(MouseEvent e) {
        if (e.getButton() == MouseButton.PRIMARY) {
            if(polyline == null) {
                polyline = new Polyline();
                polyline.setStroke(Color.LIGHTCORAL);
                polyline.setStrokeWidth(3);
                polyline.setStrokeLineCap(StrokeLineCap.ROUND);
                polyline.getPoints().addAll(e.getX(), e.getY());
                map.getChildren().add(polyline);
            }
            polyline.getPoints().addAll(e.getX(), e.getY());
        } else {
            if(polyline != null) {
                Polyline p = new Polyline();
                p.setStroke(Color.LIGHTCORAL);
                p.setStrokeWidth(3);
                p.setStrokeLineCap(StrokeLineCap.ROUND);
                p.getPoints().addAll(polyline.getPoints().subList(0, polyline.getPoints().size()-2));

                map.getChildren().remove(polyline);
                polyline = null;

                setTramLinesHandlers(p);
                // tram line added to map
                if(sem.insertTramLine(updatePolylineCoordinates(p.getPoints(),p))) {
                    map.getChildren().add(p);
                    // reference to tram line stored into hash map
                    mapNodes.put(maxId++, p);
                    inputDialogSpatialEntity = new InputDialogSpatialEntity(this);
                    inputDialogSpatialEntity.showStage();
                    setFocusedOf();
                }
                // exit insert mode
                this.mode = 0;
            }
        }
    }

    public void deleteFocusedNode() {
        if(mode == 0){
            if(focused >= 0) {
                // remove Node from map
                map.getChildren().remove(mapNodes.get(focused));
                // remove reference to removed object
                mapNodes.remove(focused);

                if(sem.getType(focused).equals("property")){
                    Picture picture = new Picture();
                    try {
                        picture.deleteOrdImagesOfSpatialEntityFromDb(focused);
                        imageSlider.loadEmptyImage();
;                    }
                    catch (SQLException ex){
                        ex.printStackTrace();
                    }
                    sem.deleteProperty(focused);
                }else {
                    Picture picture = new Picture();
                    try {
                        picture.deleteOrdImagesOfSpatialEntityFromDb(focused);
                        imageSlider.loadEmptyImage();
                    }
                    catch (SQLException ex){
                        ex.printStackTrace();
                    }
                    sem.deleteSpatialEntity(focused);
                }

                try {
                    maxId = dbm.getNewId("SpatialEntities");
                } catch (SQLException e) {}
                focused = -1;
                // remove control points if node had them
                clearControlPoints();
                setBlockLayer();
            }
        }
    }

    public Pair<Integer,Integer> getTwoClosestPoints (Point2D point, ObservableList<Double> points) {
        double d1 = map.getWidth() + 100;
        double d2 = map.getWidth() + 100;
        int i1 = 0;
        int i2 = 0;
        double x2, y2, distance;

        for(int i = 0; i < points.size(); i+=2) {
            x2 = Math.pow((points.get(i) - point.getX()), 2.0);
            y2 = Math.pow((points.get(i + 1) - point.getY()), 2.0);
            distance = Math.sqrt(x2 + y2);
            if (distance < d1) {
                d2 = d1;
                d1 = distance;
                i2 = i1;
                i1 = i;
            } else if (distance < d2) {
                d2 = distance;
                i2 = i;
            }
        }
        // return ordered pair
        if(i1 < i2) {
            return new Pair<>(i1, i2);
        } else {
            return new Pair<>(i2, i1);
        }
    }

    public void clearControlPoints() {
        if(cList.size() != 0) {
            for (Circle c: cList) { map.getChildren().remove(c); }
            cList.clear();
        }
    }

    public void addPointToTramLine(MouseEvent e) {
        Polyline road = (Polyline) mapNodes.get(focused);
        ArrayList<Double> a = new ArrayList();

        double x = e.getX() - road.getLayoutX();
        double y = e.getY() - road.getLayoutY();

        double fpx = road.getPoints().get(0);
        double fpy = road.getPoints().get(1);
        double lpx = road.getPoints().get(road.getPoints().size() - 2);
        double lpy = road.getPoints().get(road.getPoints().size() - 1);

        double distanceToFirst = Math.sqrt(((fpx - x) * (fpx - x)) + ((fpy - y) * (fpy - y)));
        double distanceToLast = Math.sqrt(((lpx - x) * (lpx - x)) + ((lpy - y) * (lpy - y)));

        if(distanceToLast < distanceToFirst) {
            a.addAll(road.getPoints());
            a.add(x);
            a.add(y);
        } else {
            a.add(0, x);
            a.add(1, y);
            a.addAll(2, road.getPoints());
        }
        if(sem.updateJGeometry(focused, updatePolylineCoordinates(a, road), sem.getType(focused))){
            road.getPoints().setAll(a);
        }
    }

    public void addPointToBuilding(MouseEvent e) {
        Polygon p = (Polygon) mapNodes.get(focused);
        double x = e.getX() - p.getLayoutX();
        double y = e.getY() - p.getLayoutY();
        ArrayList<Double> listWithNewPoint = new ArrayList();
        Pair<Integer,Integer> closest = getTwoClosestPoints(new Point2D(x, y), p.getPoints());
        int index=0;

        if(closest.getKey() == 0 && (closest.getValue() == p.getPoints().size() - 2) ) {
            listWithNewPoint.addAll(p.getPoints());
            listWithNewPoint.add(x);
            listWithNewPoint.add(y);
        }
        else {
            while (index != closest.getKey() + 2) {
                listWithNewPoint.add(index, p.getPoints().get(index));
                index++;
            }
            listWithNewPoint.add(index++, x);
            listWithNewPoint.add(index++, y);
            listWithNewPoint.addAll(index, p.getPoints().subList(index - 2, p.getPoints().size()));
        }
        Polygon normalizedPolygon = updatePolygonCoordinates(listWithNewPoint, p);
        if(sem.updateJGeometry(focused, normalizedPolygon, sem.getType(focused))){
            if(sem.checkJGeometryValidity()) {
                p.getPoints().setAll(listWithNewPoint);
            } else {
                Polygon norm = updatePolygonCoordinates(p.getPoints(), p);
                sem.updateJGeometry(focused, norm, sem.getType(focused));
            }
        }
    }

    public void modifyTramLine() {
        Polyline p = (Polyline) mapNodes.get(focused);
        if(cList.isEmpty()) {
            for(int i = 0; i < p.getPoints().size(); i+=2) {
                Circle c = new Circle(p.getPoints().get(i) + p.getLayoutX(), p.getPoints().get(i + 1) + p.getLayoutY(), 5.0);
                c.setFill(Color.YELLOW);
                c.setStrokeWidth(2);
                c.setStroke(Color.DARKGRAY);
                c.setStrokeDashOffset(2.0);
                c.setOnMousePressed(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent mouseEvent) {
                        oldLayoutCY = c.getLayoutY();
                        oldLayoutCX = c.getLayoutX();
                        Circle pressedCircle = (Circle) mouseEvent.getSource();
                        pressedCircle.toFront();
                        mousePosition.set(new Point2D(mouseEvent.getSceneX(), mouseEvent.getSceneY()));
                        if (mouseEvent.getButton() == MouseButton.MIDDLE) {
                            Polyline p = (Polyline) mapNodes.get(focused);
                            double fpx = p.getPoints().get(0);
                            double fpy = p.getPoints().get(1);
                            double lpx = p.getPoints().get(p.getPoints().size() - 2);
                            double lpy = p.getPoints().get(p.getPoints().size() - 1);

                            double cx = ((Circle)mouseEvent.getSource()).getCenterX() - p.getLayoutX();
                            double cy = ((Circle)mouseEvent.getSource()).getCenterY() - p.getLayoutY();

                            if(((cx == fpx && cy == fpy) || (cx == lpx && cy == lpy)) && p.getPoints().size() > 4) {
                                map.getChildren().remove(mouseEvent.getSource());
                                int index = cList.indexOf(mouseEvent.getSource()) * 2;
                                cList.remove(mouseEvent.getSource());
                                p.getPoints().remove(index);
                                p.getPoints().remove(index);
                                sem.updateJGeometry(focused, updatePolylineCoordinates(p.getPoints(),p), sem.getType(focused));
                            }
                        }
                    }
                });
                c.setOnMouseDragged(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent m) {
                        if(m.getButton() == MouseButton.PRIMARY) {
                            onDragNode(m, true);
                        }
                    }
                });
                c.setOnDragDetected(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent mouseEvent) {
                        wasDraged = true;
                    }
                });
                c.setOnMouseReleased(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent mouseEvent) {
                        if(mouseEvent.getButton() == MouseButton.MIDDLE) {
                            clearControlPoints();
                            modifyTramLine();
                        }
                        if(wasDraged){
                            Polyline focusedPolyline = (Polyline) mapNodes.get(focused);
                            Polyline dbPolyline = updatePolylineCoordinates(focusedPolyline.getPoints(),focusedPolyline);
                            if(!sem.updateJGeometry(focused, dbPolyline, sem.getType(focused))) {
                                int index = cList.indexOf(mouseEvent.getSource()) * 2;
                                c.setLayoutX(oldLayoutCX);
                                c.setLayoutY(oldLayoutCY);
                                focusedPolyline.getPoints().set(index, (c.getCenterX() - focusedPolyline.getLayoutX()) + oldLayoutCX);
                                focusedPolyline.getPoints().set(index+1, (c.getCenterY() - focusedPolyline.getLayoutY()) + oldLayoutY);
                            }
                        }
                    }
                });
                map.getChildren().add(c);
                cList.add(c);
            }
        }
    }

    public Polyline updatePolylineCoordinates(List<Double> points, Polyline p) {
        ArrayList<Double> list = addLayoutOffset(points, p);
        Polyline polyline = new Polyline();
        polyline.getPoints().addAll(normalizeCoordinatesDB(list));

        return polyline;
    }

    public Polygon updatePolygonCoordinates(List<Double> points, Polygon p) {
        ArrayList<Double> list = addLayoutOffset(points, p);
        Polygon shiftedPolygon = new Polygon();
        shiftedPolygon.getPoints().addAll(list);

        return normalizePolygonDB(shiftedPolygon);
    }

    public void modifyBuilding() {
        Polygon p = (Polygon)mapNodes.get(focused);
        if(cList.isEmpty()) {
            for(int i = 0; i < p.getPoints().size(); i+=2) {
                Circle c = new Circle(p.getPoints().get(i) + p.getLayoutX(), p.getPoints().get(i + 1) + p.getLayoutY(), 5.0);
                c.setFill(Color.YELLOW);
                c.setStrokeWidth(2);
                c.setStroke(Color.DARKGRAY);
                c.setStrokeDashOffset(2.0);
                c.setOnMousePressed(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent mouseEvent) {
                        oldLayoutCY = c.getLayoutY();
                        oldLayoutCX = c.getLayoutX();
                        Circle pressedCircle = (Circle) mouseEvent.getSource();
                        pressedCircle.toFront();
                        mousePosition.set(new Point2D(mouseEvent.getSceneX(), mouseEvent.getSceneY()));
                        if (mouseEvent.getButton() == MouseButton.MIDDLE) {
                            Polygon p = (Polygon) mapNodes.get(focused);
                            if (p.getPoints().size() / 2 > 3) {
                                map.getChildren().remove(mouseEvent.getSource());
                                int index = cList.indexOf(mouseEvent.getSource()) * 2;
                                cList.remove(mouseEvent.getSource());

                                ArrayList<Double> oldPoints = new ArrayList<>();
                                oldPoints.addAll(p.getPoints());

                                p.getPoints().remove(index);
                                p.getPoints().remove(index);
                                Polygon updated = updatePolygonCoordinates(p.getPoints(), p);
                                if(!sem.updateJGeometry(focused, updated, sem.getType(focused))) {
                                    Polygon old = updatePolygonCoordinates(oldPoints, p);
                                    sem.updateJGeometry(focused, old, sem.getType(focused));
                                    p.getPoints().clear();
                                    p.getPoints().addAll(oldPoints);
                                    clearControlPoints();
                                    modifyBuilding();
                                }
                            }
                        }
                    }
                });
                c.setOnMouseDragged(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent m) {
                        if(m.getButton() == MouseButton.PRIMARY) {
                            onDragNode(m, true);
                        }
                    }
                });
                c.setOnDragDetected(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent mouseEvent) {
                        wasDraged = true;
                    }
                });
                c.setOnMouseReleased(new EventHandler<MouseEvent>() {
                    @Override
                    public void handle(MouseEvent mouseEvent) {
                        if(wasDraged) {
                            wasDraged = false;
                            Polygon focusedPolygon = (Polygon) mapNodes.get(focused);
                            Polygon normalizedPolygon = updatePolygonCoordinates(focusedPolygon.getPoints(), focusedPolygon);
                            if (sem.updateJGeometry(focused, normalizedPolygon, sem.getType(focused))) {
                                if (!sem.checkJGeometryValidity()) {
                                    int index = cList.indexOf(mouseEvent.getSource()) * 2;
                                    c.setLayoutX(oldLayoutCX);
                                    c.setLayoutY(oldLayoutCY);
                                    focusedPolygon.getPoints().set(index, (c.getCenterX() - focusedPolygon.getLayoutX()) + oldLayoutCX);
                                    focusedPolygon.getPoints().set(index + 1, (c.getCenterY() - focusedPolygon.getLayoutY()) + oldLayoutCY);
                                    sem.updateJGeometry(focused, updatePolygonCoordinates(focusedPolygon.getPoints(), focusedPolygon), sem.getType(focused));
                                }
                            } else {
                                int index = cList.indexOf(mouseEvent.getSource()) * 2;
                                c.setLayoutX(oldLayoutCX);
                                c.setLayoutY(oldLayoutCY);
                                focusedPolygon.getPoints().set(index, (c.getCenterX() - focusedPolygon.getLayoutX()) + oldLayoutCX);
                                focusedPolygon.getPoints().set(index + 1, (c.getCenterY() - focusedPolygon.getLayoutY()) + oldLayoutCY);
                            }
                        }
                    }
                });
                map.getChildren().add(c);
                cList.add(c);
            }
        }
    }

    public double[] groupToArray(Group group, double layoutX, double layoutY) {
        int groupSize = group.getChildren().size();
        double[] databaseTrees = new double[(groupSize*2)];

        for (int i = 0; i < groupSize; i++) {
            databaseTrees[i*2] = ((Circle)group.getChildren().get(i)).getCenterX() + layoutX;
            databaseTrees[i*2+1] = 600 - (((Circle)group.getChildren().get(i)).getCenterY() + layoutY);
        }

        return databaseTrees;
    }

    public void createTreesCollection(MouseEvent e) {
        if(e.getButton() == MouseButton.PRIMARY) {
            double x = e.getX();
            double y = e.getY();

            Circle c = new Circle(x, y, 8.0);
            c.setFill(Color.LIGHTGREEN);
            c.setStroke(Color.DARKGREEN);
            c.setStrokeWidth(3.0);
            setTreeHandler(c);
            trees.add(c);
            map.getChildren().add(c);
        } else {
            Group group = new Group();
            group.getChildren().addAll(trees);
            setMultipointHandlers(group);

            if(sem.insertTrees(groupToArray(group, 0.0, 0.0))) {
                mapNodes.put(maxId++, group);
                trees.clear();
                map.getChildren().add(group);
                inputDialogSpatialEntity = new InputDialogSpatialEntity(this);
                inputDialogSpatialEntity.showStage();
                setFocusedOf();
            }

            this.mode = 0;
        }
    }

    public void addTreeToGroup(MouseEvent e) {
        Group g = (Group)mapNodes.get(focused);
        double x = e.getX() - g.getLayoutX();
        double y = e.getY() - g.getLayoutY();

        Circle c = new Circle(x, y, 8.0);
        c.setFill(Color.LIGHTGREEN);
        c.setStroke(Color.DARKGREEN);
        c.setStrokeWidth(3.0);
        setTreeHandler(c);
        g.getChildren().add(c);

        if(!sem.updateJGeometry(focused, groupToArray(g, g.getLayoutX(), g.getLayoutY()), sem.getType(focused))){
            g.getChildren().remove(c);
            sem.updateJGeometry(focused, groupToArray(g, g.getLayoutX(), g.getLayoutY()), sem.getType(focused));
        }
    }

    @FXML
    public void onMapClicked(MouseEvent e) {
        if (mode == 1) { createBuilding(e); }
        else if (mode == 2) { createTramLine(e); }
        else if (mode == 3) { createTramStop(e.getX(), e.getY(), -1); }
        else if (mode == 4) { createTreesCollection(e); }
        else if (mapNodes.get(focused) instanceof Group && e.getButton() == MouseButton.MIDDLE) { addTreeToGroup(e); }
        else if (mapNodes.get(focused) instanceof Polygon && e.getButton() == MouseButton.SECONDARY) { modifyBuilding(); }
        else if (mapNodes.get(focused) instanceof Polyline && e.getButton() == MouseButton.SECONDARY) { modifyTramLine(); }
        else if (mapNodes.get(focused) instanceof Polygon && e.getButton() == MouseButton.MIDDLE) {
            addPointToBuilding(e);
            clearControlPoints();
            modifyBuilding();
        }   else if (mapNodes.get(focused) instanceof Polyline && e.getButton() == MouseButton.MIDDLE) {
            addPointToTramLine(e);
            clearControlPoints();
            modifyTramLine();
        }
    }
}
