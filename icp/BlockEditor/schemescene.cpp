/**
* @file schemescene.cpp
* @brief SchemeScene implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "schemescene.h"
#include "link.h"

#include <QTextCursor>
#include <QtMath>
#include <QGraphicsView>
#include <QGraphicsSceneMouseEvent>
#include <QToolTip>
#include <QGraphicsItem>
#include <QInputDialog>
#include <QMessageBox>

/**
 * @brief SchemeScene constructor
 * @param parent
 */
SchemeScene::SchemeScene(QObject *parent): QGraphicsScene(parent)
{
    scheme = new Scheme();
    scheme->generateNewId();
    sceneMode = MoveItem;
    sceneItemType = SchemeItem::Step;
    line = 0;
}

/**
 * @brief Set mode of scene
 * @param mode
 */
void SchemeScene::setMode(Mode mode)
{
    sceneMode = mode;
}

/**
 * @brief Set type of item in scene
 * @param type type which will be set
 */
void SchemeScene::setItemType(SchemeItem::DiagramType type)
{
    sceneItemType = type;
}

/**
 * @brief Overriden mousePressEvent method. If mode of scene is InsterItem mode
 *        new item will be added to scheme to mousePressEvent position. If mode of
 *        scene is InsertLine field line is set to actual position (both start and end point)
 * @param mouseEvent
 */
void SchemeScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if (mouseEvent->button() != Qt::LeftButton){
        return;
     }

    SchemeItem *item;
    switch (sceneMode) {
        case InsertItem:
            item = new SchemeItem(sceneItemType);
            addItem(item);
            item->setPos(mouseEvent->scenePos());
            item->block->item = item;
            scheme->addBlock(item->block);
            item->setToolTip(QString::fromStdString(item->block->portsInfo()));
            emit itemInserted(item);
            break;
        case InsertLine:
            line = new QGraphicsLineItem(QLineF(mouseEvent->scenePos(),
                                        mouseEvent->scenePos()));
            addItem(line);
            break;
        default:
        ;
    }
    QGraphicsScene::mousePressEvent(mouseEvent);
}

/**
 * @brief Overriden mouseMoveEvent. If mode of scene is MoveItem, parent methos is called.
 *        If mode of scene is InsertLine field line is set to new line with start point at
 *        position of mousePressEvent and end point at actual mouse poistion.
 * @param mouseEvent
 */
void SchemeScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    if (sceneMode == InsertLine && line != 0) {
        QLineF newLine(line->line().p1(), mouseEvent->scenePos());
        line->setLine(newLine);
    } else if (sceneMode == MoveItem) {
        QGraphicsScene::mouseMoveEvent(mouseEvent);
    }
}

/**
 * @brief Overriden method mouseReleaseEvent. If mode of scene is InsertLine, we must
 *        check if out line start at one block and end in another. In case that this condition
 *        is valid line is inserted into scene.
 * @param mouseEvent
 */
void SchemeScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    SchemeItem *startItem = NULL;
    SchemeItem *endItem = NULL;
    if (line != 0 && sceneMode == InsertLine) {
        QList<QGraphicsItem *> startItems = items(line->line().p1());
        if (startItems.count() && startItems.first() == line)
            startItems.removeFirst();
        QList<QGraphicsItem *> endItems = items(line->line().p2());
        if (endItems.count() && endItems.first() == line)
            endItems.removeFirst();

        removeItem(line);
        delete line;

        if (startItems.count() > 0 && endItems.count() > 0 &&
            startItems.first()->type() == SchemeItem::Type &&
            endItems.first()->type() == SchemeItem::Type &&
            startItems.first() != endItems.first()) {
            startItem = qgraphicsitem_cast<SchemeItem *>(startItems.first());
            endItem = qgraphicsitem_cast<SchemeItem *>(endItems.first());
            Link *link;
            if(!startItem->output1 && (endItem->input1 == 0 || endItem->input2==0)){
                if(!endItem->input1 && endItem->input2 != -1){
                    link = new Link(startItem, endItem, QPoint(-15,0));
                    endItem->input1 = 1;
                    startItem->output1 = 1;
                    link->connection = new Connection(&startItem->block->outPorts[0],&endItem->block->inPorts[0]);
                }else if(!endItem->input1 && endItem->input2 == -1){
                    link = new Link(startItem, endItem, QPoint(-15,15));
                    endItem->input1 = 1;
                    startItem->output1 = 1;
                    link->connection = new Connection(&startItem->block->outPorts[0],&endItem->block->inPorts[0]);
                }else if(!endItem->input2){
                    link = new Link(startItem, endItem, QPoint(-15,30));
                    endItem->input2 = 2;
                    startItem->output1 = 2;
                    link->connection = new Connection(&startItem->block->outPorts[0],&endItem->block->inPorts[1]);
                }
                scheme->addConnection(link->connection);
                int cycleErr, compatibleErr;
                cycleErr = scheme->checkForCycle();
                compatibleErr = link->connection->checkCompatibility();
                if(cycleErr == -1 || compatibleErr == -1){
                    switch (cycleErr) {
                    case 0:
                        emit compatibleError();
                        break;
                    case -1:
                        emit cycleError();
                        break;
                    }
                    scheme->removeConnection(link->connection);
                    switch (startItem->output1) {
                     case 1:
                         endItem->input1 = 0;
                         break;
                     case 2:
                         endItem->input2 = 0;
                         break;
                    }
                    startItem->output1 = 0;
                    delete link;
                }else{
                    link->setToolTip(QString::fromStdString(link->connection->input->mapToString()));
                    startItem->addLink(link);
                    endItem->addLink(link);
                    link->setZValue(-1000.0);
                    addItem(link);
                    link->updatePosition();
                }
            }
        }
    }
    line = 0;
    QGraphicsScene::mouseReleaseEvent(mouseEvent);
}

/**
 * @brief Update items tooltips (lines and blocks)
 */
void SchemeScene::updateToolTips(){
    QList<QGraphicsItem*>it;
    it = this->items();
    for(int i = 0; i<it.size(); i++){
        Link *line_at = qgraphicsitem_cast<Link *>(it.value(i));
        if(line_at == nullptr){
            SchemeItem *item_at = qgraphicsitem_cast<SchemeItem *>(it.value(i));
            item_at->setToolTip(QString::fromStdString(item_at->block->portsInfo()));
        }else{
            line_at->setToolTip(QString::fromStdString(line_at->connection->input->mapToString()));
        }
    }
}

/**
 * @brief Remove all items from scene.
 */
void SchemeScene::clearAll(){
    QList<QGraphicsItem*>it;
    it = this->items();
    for(int i = 0; i<it.size(); i++){
        this->removeItem(it.value(i));
    }
}

/**
 * @brief Update positions of all blocks in scene and save it to position field of each block.
 */
void SchemeScene::updatePositions(){
    QList<QGraphicsItem*>it;
    it = this->items();
    for(int i = 0; i<it.size(); i++){
        SchemeItem *item_at = qgraphicsitem_cast<SchemeItem *>(it.value(i));
        if(item_at == nullptr)
            continue;
        item_at->block->setPosition((item_at->scenePos().toPoint().x()),(item_at->scenePos().toPoint().y()));
    }
}
