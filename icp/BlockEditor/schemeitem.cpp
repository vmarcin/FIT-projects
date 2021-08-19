/**
* @file schemeitem.cpp
* @brief SchemeItem implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "schemeitem.h"
#include "link.h"

#include <QGraphicsScene>
#include <QMenu>
#include <QPainter>
#include <QStyleOption>
#include <QGraphicsSceneMouseEvent>

/**
 * @brief SchemeItem constructor
 * @param diagramType type of item in scene
 * @param parent
 */
SchemeItem::SchemeItem(DiagramType sceneItemType, QGraphicsItem *parent) : QGraphicsItem(parent)
{
    input1 = 0;
    input2 = 0;
    output1 = 0;

    itemType = sceneItemType;

    switch (itemType) {
    case DiagramType::Add:
        block = new Adder();
        break;
    case DiagramType::Sub:
        block = new Subtractor();
        break;
    case DiagramType::Mul:
        block = new Multiplier();
        break;
    case DiagramType::Div:
        block = new Divider();
        break;
    case DiagramType::GR:
        block = new GetReal();
        break;
    case DiagramType::GI:
        block = new GetImaginary();
        break;
    case DiagramType::GA:
        block = new PolarForm();
        break;
    case DiagramType::CC:
        block = new CreateComplex();
        break;
    case DiagramType::CP:
        block = new PolarToComplex();
        break;
    default:
        break;
    }

    if(sceneItemType == 4 || sceneItemType == 5 || sceneItemType == 6 || sceneItemType == 8)
        input2 = -1;
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
}

/**
 * @brief Compute boundict rectangle of block
 * @return return computed rectangle
 */
QRectF SchemeItem::boundingRect() const
{
     qreal penWidth = 1;
     return QRectF(-15 - penWidth, -10 - penWidth,
                   70 + penWidth, 50 + penWidth);
}

/**
 * @brief Paint block to scene
 * @param painter
 */
void SchemeItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{
    if(isSelected())
        painter->setPen(QPen(Qt::darkGray, 2, Qt::DashLine));
    else
        painter->setPen(QPen(Qt::black, 2));
    painter->setBrush(Qt::white);
    painter->drawRect(-10,-10,60,50);
    painter->setBrush(Qt::darkBlue);
    switch (itemType) {
    case 0:
        painter->drawText(7,20,"ADD");
        painter->drawEllipse(-15,-5,10,10);
        painter->drawEllipse(-15,25,10,10);
        break;
    case 1:
        painter->drawText(7,20,"SUB");
        painter->drawEllipse(-15,-5,10,10);
        painter->drawEllipse(-15,25,10,10);
        break;
    case 2:
        painter->drawText(7,20,"MUL");
        painter->drawEllipse(-15,-5,10,10);
        painter->drawEllipse(-15,25,10,10);
        break;
    case 3:
        painter->drawText(7,20,"DIV");
        painter->drawEllipse(-15,-5,10,10);
        painter->drawEllipse(-15,25,10,10);
        break;
    case 4:
        painter->drawText(10,20,"Re");
        painter->drawEllipse(-15,10,10,10);
        break;
    case 5:
        painter->drawText(10,20,"Im");
        painter->drawEllipse(-15,10,10,10);
        break;
    case 6:
        painter->drawText(-1,20,"POLAR");
        painter->drawEllipse(-15,10,10,10);
        break;
    case 7:
        painter->drawText(0,20,"COMP");
        painter->drawEllipse(-15,-5,10,10);
        painter->drawEllipse(-15,25,10,10);
        break;
    case 8:
        painter->drawText(0,20,"COMP");
        painter->drawEllipse(-15,10,10,10);
        break;
    default:
        break;
    }
    painter->drawEllipse(45,10, 10,10);
}

/**
 * @brief Remove link from scene item
 * @param link link which you want delete
 */
void SchemeItem::removeLink(Link *link)
{
    int index = links.indexOf(link);
    switch (link->getSourceItem()->output1) {
    case 1:
        link->getDestItem()->input1 = 0;
        break;
    case 2:
        link->getDestItem()->input2 = 0;
        break;
    }
    link->getSourceItem()->output1 = 0;
    if (index != -1)
        links.removeAt(index);
}

/**
 * @brief Remove all links connected to scene item
 */
void SchemeItem::removeLinks()
{
    foreach (Link *link, links) {
        link->getSourceItem()->removeLink(link);
        link->getDestItem()->removeLink(link);
        scene()->removeItem(link);
        delete link;
    }
}

/**
 * @brief Add item to scene item
 * @param link link wich you want to add to item
 */
void SchemeItem::addLink(Link *link)
{
    links.append(link);
}

/**
 * @brief Update position of links connected to block which change its position
 * @param change
 * @param value
 * @return
 */
QVariant SchemeItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemPositionChange) {
        foreach (Link *link, links) {
            link->updatePosition();
        }
    }

    return value;
}
