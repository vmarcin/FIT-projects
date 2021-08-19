/**
* @file mainwindow.h
* @brief Mainwindow interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "schemeitem.h"

#include <QMainWindow>
#include <QButtonGroup>
#include <QToolBox>
#include <QGraphicsView>

class SchemeScene;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
   MainWindow();

private slots:
    void operationGroupClicked(int id);
    void deleteItem();
    void pointerGroupClicked(int id);
    void itemInserted(SchemeItem *item);
    void cycleError();
    void compatibleError();
    void about();
    void saveScheme();
    void newScheme();
    void loadScheme();
    void step();
    void run();
    void exit();

private:
    int askForValues();
    void createToolBox();
    void createActions();
    void createMenus();
    void createToolbars();
    QWidget *createCellWidget(const QString &text, SchemeItem::DiagramType type);

    SchemeScene *scene;
    QGraphicsView *view;

    QAction *exitAction;
    QAction *addAction;
    QAction *deleteAction;
    QAction *aboutAction;
    QAction *stepAction;
    QAction *runAction;
    QAction *saveAction;
    QAction *loadAction;
    QAction *newAction;

    QMenu *fileMenu;
    QMenu *aboutMenu;

    QToolBar *editToolBar;
    QToolBar *pointerToolbar;
    QToolBar *stepingToolbar;

    QToolBox *toolBox;

    QButtonGroup *operationsGroup;
    QButtonGroup *pointersGroup;
};

#endif // MAINWINDOW_H
