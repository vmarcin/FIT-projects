/**
* @file mainwindow.cpp
* @brief MainWindow implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "link.h"
#include "schemeitem.h"
#include "schemescene.h"
#include "mainwindow.h"

#include <QtWidgets>
#include <iostream>
#include <fstream>

/**
 * @brief MainWindow constructor
 */
MainWindow::MainWindow()
{
    createActions();
    createToolBox();
    createMenus();
    createToolbars();

    scene = new SchemeScene(this);
    scene->setSceneRect(QRectF(0, 0, 5000, 5000));
    connect(scene, SIGNAL(itemInserted(SchemeItem*)), this, SLOT(itemInserted(SchemeItem*)));
    connect(scene, SIGNAL(compatibleError()), this, SLOT(compatibleError()));
    connect(scene, SIGNAL(cycleError()), this, SLOT(cycleError()));

    QHBoxLayout *layout = new QHBoxLayout;
    layout->addWidget(toolBox);
    view = new QGraphicsView(scene);
    view->setMouseTracking(true);
    layout->addWidget(view);

    QWidget *widget = new QWidget;
    widget->setLayout(layout);

    setCentralWidget(widget);
    setWindowTitle(tr("BlockEditor"));
    setUnifiedTitleAndToolBarOnMac(true);
}

/**
 * @brief When a button in operationsGroup is clicked
 *        the mode of scene will be switch into insert mode
 *        and according to button item type is set
 * @param id id of button used for item type identification
 */
void MainWindow::operationGroupClicked(int id)
{
    QList<QAbstractButton *> buttons = operationsGroup->buttons();
    foreach (QAbstractButton *button, buttons) {
        if (operationsGroup->button(id) != button)
            button->setChecked(false);
    }
    scene->setItemType(SchemeItem::DiagramType(id));
    scene->setMode(SchemeScene::InsertItem);
}

/**
 * @brief Delete selected items from scene. If item is a block
 *        we need also remove all connections to it.
 */
void MainWindow::deleteItem()
{
    foreach (QGraphicsItem *item, scene->selectedItems()) {
        if (item->type() == Link::Type) {
            scene->removeItem(item);
            Link *link = qgraphicsitem_cast<Link *>(item);
            scene->scheme->removeConnection(link->connection);
            link->getSourceItem()->removeLink(link);
            link->getDestItem()->removeLink(link);
            delete item;
        }
    }

    foreach (QGraphicsItem *item, scene->selectedItems()) {
         if (item->type() == SchemeItem::Type){
             SchemeItem* b = qgraphicsitem_cast<SchemeItem *>(item);
             b->removeLinks();
             scene->scheme->removeBlock(b->block);
         }
         scene->removeItem(item);
         delete item;
     }
}

/**
 * @brief Set mode according to pressed button from pointersGroup
 */
void MainWindow::pointerGroupClicked(int)
{
    scene->setMode(SchemeScene::Mode(pointersGroup->checkedId()));
}

/**
 * @brief MainWindow::itemInserted
 * @param item
 */
void MainWindow::itemInserted(SchemeItem *item)
{
    pointersGroup->button(int(SchemeScene::MoveItem))->setChecked(true);
    scene->setMode(SchemeScene::Mode(pointersGroup->checkedId()));
    operationsGroup->button(int(item->diagramType()))->setChecked(false);
}

/**
 * @brief Prints about message of application
 */
void MainWindow::about()
{
    QMessageBox::about(this, tr("About Block Editor"),
                       tr("The <b>Block Editor</b> offers "
                          "basic operations with complex numbers."));
}

/**
 * @brief Prints cycle error message
 */
void MainWindow::cycleError(){
    QMessageBox::warning(this, tr("ERROR") ,tr("<b>Unable to connect</b> (a cycle will occur after connection)."));
}

/**
 * @brief Prints compatible error message
 */
void MainWindow::compatibleError(){
    QMessageBox::warning(this, tr("ERROR"),tr("<b>Unable to connect</b> (ports are not compatible)."));
}

/**
 * @brief Asks input values from user
 * @return returns 0 if all values are given -1 otherwise
 */
int MainWindow::askForValues()
{
    QList<QGraphicsItem*>it;
    it = scene->items();
    bool ok, ok1;
    double real, imaginary;
    double magnitude, angle;
    for(int i = it.size()- 1; i >= 0; i--){
        SchemeItem *item_at = qgraphicsitem_cast<SchemeItem*>(it.value(i));
        if(item_at == nullptr)
            continue;
        item_at->setSelected(true);
        int blockType = item_at->diagramType();
        if(item_at->input1 == 0){
            if(blockType == 7){
                real = QInputDialog::getDouble(this, tr("Set Real"),tr("<b>input1</b>\n  real:"),5.5,-10000,10000,5,&ok);
                if(ok)
                    item_at->block->inPorts[0].setReal(real);
                else
                    return -1;
            }else if(blockType == 8){
                magnitude = QInputDialog::getDouble(this, tr("Set Magnitude"),tr("<b>input1</b>\n  magnitude:"),5.5,-10000,10000,5,&ok);
                angle = QInputDialog::getDouble(this, tr("Set Angle"),tr("<b>input1</b>\n  angle(rad):"),5.5,-10000,10000,5,&ok1);
                if(ok && ok1)
                    item_at->block->inPorts[0].setPolar(magnitude,angle);
                else
                    return -1;
            }else{
                real = QInputDialog::getDouble(this, tr("Set Real"),tr("<b>input1</b>\n  real:"),5.5,-10000,10000,5,&ok);
                imaginary = QInputDialog::getDouble(this, tr("Set Imaginary"),tr("<b>input1</b>\n  imaginary:"),5.5,-10000,10000,5,&ok1);
                if(ok && ok1)
                    item_at->block->inPorts[0].setComplex(real,imaginary);
                else
                    return -1;
            }
        }
        if(item_at->input2 == 0){
            if(blockType == 7){
                imaginary = QInputDialog::getDouble(this, tr("Set Imaginary"),tr("<b>input2</b>\n  imaginary:"),5.5,-10000,10000,5,&ok);
                if(ok)
                    item_at->block->inPorts[1].setImaginary(imaginary);
                else
                    return -1;
            }else{
                real = QInputDialog::getDouble(this, tr("Set Real"),tr("<b>input2</b>\n  real:"),5.5,-10000,10000,5,&ok);
                imaginary = QInputDialog::getDouble(this, tr("Set Imaginary"),tr("<b>input2</b>\n  imaginary:"),5.5,-10000,10000,5,&ok1);
                if(ok && ok1)
                    item_at->block->inPorts[1].setComplex(real,imaginary);
                else return -1;
            }
        }
        item_at->setSelected(false);
    }
    return 0;
}

/**
 * @brief Called when stepAction occures. Performs one step of computation
 */
void MainWindow::step()
{
    static bool cnt = false;
    static SchemeItem* prev_item = nullptr;
    if(!cnt){
        scene->scheme->clearPorts();
        if(askForValues() == -1)
            return;
        cnt = true;
    }
    if(prev_item != nullptr)
        prev_item->setSelected(false);
    SchemeItem *it = scene->scheme->run(false);
    scene->updateToolTips();
    if(it != nullptr){
        prev_item = it;
        it->setSelected(true);
    }
    else{
        cnt = false;
        prev_item = nullptr;
    }
}

/**
 * @brief Called when runAction occures. Performs computation
 */
void MainWindow::run()
{
    scene->scheme->clearPorts();
    if(askForValues() == -1)
        return;
    scene->scheme->run(true);
    scene->updateToolTips();
}

/**
 * @brief Saves scheme into given file
 */
void MainWindow::saveScheme(){
    std::string schemeId = "Scheme";
    schemeId.append(std::to_string(scene->scheme->getId()));
    QString fileName = QFileDialog::getSaveFileName(this,tr("Save scheme to file"),QString::fromStdString(schemeId));
    std::cout << fileName.toStdString() << std::endl;
    if(fileName.isEmpty())
        return;
    else{
        QFile file(fileName);
        if(!file.open(QIODevice::WriteOnly)){
            QMessageBox::information(this, tr("Unable open file"), file.errorString());
        }else{
            scene->updatePositions();
            scene->scheme->saveScheme(fileName.toStdString());
        }
    }
}

/**
 * @brief Clear actual and create new scheme
 */
void MainWindow::newScheme(){
    scene->scheme->generateNewId();
    if(scene->items().size() != 0){
        QMessageBox msgBox;
        msgBox.setText(tr("<b>The document has been modified.</b>"));
        msgBox.setInformativeText("  Do you want to save your changes?");
        msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
        msgBox.setDefaultButton(QMessageBox::Save);
        int ret = msgBox.exec();
        switch (ret) {
          case QMessageBox::Save:
              saveScheme();
              break;
          case QMessageBox::Discard:
              scene->scheme->clearScheme();
              scene->clearAll();
              break;
          case QMessageBox::Cancel:
              break;
        }
    }
}

/**
 * @brief Exit application and ask user if he want to save changes if some exists
 */
void MainWindow::exit(){
    if(scene->items().size() != 0){
        QMessageBox msgBox;
        msgBox.setText(tr("<b>The document has been modified.</b>"));
        msgBox.setInformativeText("  Do you want to save your changes?");
        msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
        msgBox.setDefaultButton(QMessageBox::Save);
        int ret = msgBox.exec();
        switch (ret) {
          case QMessageBox::Save:
              saveScheme();
              break;
          case QMessageBox::Discard:
              close();
              break;
          case QMessageBox::Cancel:
              break;
        }
    }else{
        close();
    }
}

/**
 * @brief Load new scheme from given file
 */
void MainWindow::loadScheme(){
    newScheme();
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open new scheme"));
    if(fileName.isEmpty())
        return;
    else{
        QFile file(fileName);
        if(!file.open(QIODevice::ReadOnly)){
            QMessageBox::information(this, tr("Unable to open file"), file.errorString());
            return;
        }
    }
    std::cout << fileName.toStdString() << std::endl;
    std::ifstream file(fileName.toStdString());
    std::string line;
    int item_type, x, y;
    int item1, item2, pos;
    //items
    SchemeItem * item;
    while(std::getline(file,line)) {
        if(line.find("#") != std::string::npos){
            break;
        }
        std::istringstream line_stream(line);
        line_stream >> item_type >> x >> y;
        QPoint point(x,y);
        item = new SchemeItem(SchemeItem::DiagramType(item_type));
        scene->addItem(item);
        item->setPos(QPointF(point));
        item->block->item = item;
        scene->scheme->addBlock(item->block);
        item->setToolTip(QString::fromStdString(item->block->portsInfo()));
    }
    Link *link;
    while(std::getline(file,line)) {
        std::istringstream line_stream(line);
        line_stream >> item1 >> item2 >> pos;
        SchemeItem *startItem = scene->scheme->Blocks[item1]->item;
        SchemeItem *endItem = scene->scheme->Blocks[item2]->item;
        link = new Link(startItem,endItem, QPoint(-15,pos));
        if(pos == 30){
            startItem->output1=2;
            endItem->input2=2;
            link->connection = new Connection(&startItem->block->outPorts[0], &endItem->block->inPorts[1]);
        }else{
            link->connection = new Connection(&startItem->block->outPorts[0], &endItem->block->inPorts[0]);
            startItem->output1=1;
            endItem->input1=1;
        }
        scene->scheme->addConnection(link->connection);
        link->setToolTip(QString::fromStdString(link->connection->input->mapToString()));
        startItem->addLink(link);
        endItem->addLink(link);
        link->setZValue(-1000.0);
        scene->addItem(link);
        link->updatePosition();
    }
}

/**
 * @brief Creates toolbox with buttons which means operations in our editor
 */
void MainWindow::createToolBox()
{
    operationsGroup = new QButtonGroup(this);
    operationsGroup->setExclusive(false);
    connect(operationsGroup, SIGNAL(buttonClicked(int)),
            this, SLOT(operationGroupClicked(int)));
    QGridLayout *layout = new QGridLayout;
    layout->addWidget(createCellWidget(tr("Adder"), SchemeItem::Add), 0, 0);
    layout->addWidget(createCellWidget(tr("Subtractor"), SchemeItem::Sub),0, 1);
    layout->addWidget(createCellWidget(tr("Multiplier"), SchemeItem::Mul), 1, 0);
    layout->addWidget(createCellWidget(tr("Divider"), SchemeItem::Div), 1, 1);
    layout->addWidget(createCellWidget(tr("Get real"), SchemeItem::GR), 2, 0);
    layout->addWidget(createCellWidget(tr("Get imaginary"), SchemeItem::GI), 2, 1);
    layout->addWidget(createCellWidget(tr("Polar form"), SchemeItem::GA), 3, 0);
    layout->addWidget(createCellWidget(tr("Complex form"), SchemeItem::CC), 3, 1);
    layout->addWidget(createCellWidget(tr("Complex form"), SchemeItem::CP), 4, 0);

    layout->setRowStretch(5, 5);
    layout->setColumnStretch(2, 5);

    QWidget *itemWidget = new QWidget;
    itemWidget->setLayout(layout);

    toolBox = new QToolBox;
    toolBox->setSizePolicy(QSizePolicy(QSizePolicy::Maximum, QSizePolicy::Ignored));
    toolBox->setMinimumWidth(itemWidget->sizeHint().width());
    toolBox->addItem(itemWidget, tr("Operations"));

}

/**
 * @brief Create actions
 */
void MainWindow::createActions()
{
    deleteAction = new QAction(QIcon(":/images/delete.png"), tr("&Delete"), this);
    deleteAction->setShortcut(tr("Delete"));
    deleteAction->setStatusTip(tr("Delete item from diagram"));
    connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteItem()));

    exitAction = new QAction(tr("&Quit"), this);
    exitAction->setShortcuts(QKeySequence::Quit);
    exitAction->setStatusTip(tr("Quit Scenediagram example"));
    connect(exitAction, SIGNAL(triggered()), this, SLOT(exit()));

    newAction = new QAction(tr("&New"), this);
    newAction->setShortcut(QKeySequence::New);
    newAction->setStatusTip(tr("Create new scheme"));
    connect(newAction, SIGNAL(triggered()), this, SLOT(newScheme()));

    saveAction = new QAction(tr("&Save"), this);
    saveAction->setShortcuts(QKeySequence::Save);
    connect(saveAction, SIGNAL(triggered()), this, SLOT(saveScheme()));

    loadAction = new QAction(tr("&Open"), this);
    loadAction->setShortcuts(QKeySequence::Open);
    connect(loadAction, SIGNAL(triggered()), this, SLOT(loadScheme()));

    aboutAction = new QAction(tr("A&bout"), this);
    aboutAction->setShortcut(tr("Ctrl+B"));
    connect(aboutAction, SIGNAL(triggered()), this, SLOT(about()));

    stepAction = new QAction(QIcon(":/images/step.png"), tr("&Step"), this);
    stepAction->setShortcut(tr("Step"));
    connect(stepAction, SIGNAL(triggered()), this, SLOT(step()));

    runAction = new QAction(QIcon(":/images/run.png"), tr("&Run"), this);
    runAction->setShortcut(tr("Run"));
    connect(runAction, SIGNAL(triggered()), this, SLOT(run()));

}

/**
 * @brief Create file and about menus
 */
void MainWindow::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(newAction);
    fileMenu->addAction(saveAction);
    fileMenu->addAction(loadAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    aboutMenu = menuBar()->addMenu(tr("&Help"));
    aboutMenu->addAction(aboutAction);
}

/**
 * @brief create toolbars
 */
void MainWindow::createToolbars()
{
    editToolBar = addToolBar(tr("Edit"));
    editToolBar->addAction(deleteAction);

    stepingToolbar = addToolBar(tr("Steping"));
    stepingToolbar->addAction(runAction);
    stepingToolbar->addAction(stepAction);

    QToolButton *pointerButton = new QToolButton;
    pointerButton->setCheckable(true);
    pointerButton->setChecked(true);
    pointerButton->setIcon(QIcon(":/images/pointer.png"));

    QToolButton *linePointerButton = new QToolButton;
    linePointerButton->setCheckable(true);
    linePointerButton->setIcon(QIcon(":/images/linepointer.png"));

    pointersGroup = new QButtonGroup(this);
    pointersGroup->addButton(pointerButton, int(SchemeScene::MoveItem));
    pointersGroup->addButton(linePointerButton, int(SchemeScene::InsertLine));
    connect(pointersGroup, SIGNAL(buttonClicked(int)),
            this, SLOT(pointerGroupClicked(int)));

    pointerToolbar = addToolBar(tr("Pointer type"));
    pointerToolbar->addWidget(pointerButton);
    pointerToolbar->addWidget(linePointerButton);
}

/**
 * @brief Creates QWidget containing a tool button and a label
 * @param text text which will be shown in label under button
 * @param type type of button (what operation will be represented)
 * @return returns created widger
 */
QWidget *MainWindow::createCellWidget(const QString &text, SchemeItem::DiagramType type)
{
    QIcon icon;
    QString string;
    switch (type) {
    case SchemeItem::DiagramType::Add:
        icon.addFile(":/images/plus.png");
        string = QString("Add two complex number");
        break;
    case SchemeItem::DiagramType::Sub:
        icon.addFile(":/images/minus.png");
        string = QString("Subract two complex number");
        break;
    case SchemeItem::DiagramType::Mul:
        icon.addFile(":/images/krat.png");
        string = QString("Multiply two complex number");
        break;
    case SchemeItem::DiagramType::Div:
        icon.addFile(":/images/div.png");
        string = QString("Divide two complex number");
        break;
    case SchemeItem::DiagramType::GR:
        icon.addFile(":/images/r.png");
        string = QString("Get only real part of complex number");
        break;
    case SchemeItem::DiagramType::GI:
        icon.addFile(":/images/i.png");
        string = QString("Get only imaginary part of complex number");
        break;
    case SchemeItem::DiagramType::GA:
        icon.addFile(":/images/polar.png");
        string = QString("Compute polar form of complex number");
        break;
    case SchemeItem::DiagramType::CC:
        icon.addFile(":/images/c.png");
        string = QString("Create complex number from real and imaginary part");
        break;
    case SchemeItem::DiagramType::CP:
        icon.addFile(":/images/c.png");
        string = QString("Create complex number from polar form");
        break;
    default:
        break;
    }

    QToolButton *button = new QToolButton;
    button->setIcon(icon);
    button->setIconSize(QSize(50, 50));
    button->setToolTip(string);
    button->setCheckable(true);
    operationsGroup->addButton(button, int(type));

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(button, 0, 0, Qt::AlignHCenter);
    layout->addWidget(new QLabel(text), 1, 0, Qt::AlignCenter);

    QWidget *widget = new QWidget;
    widget->setLayout(layout);

    return widget;
}
