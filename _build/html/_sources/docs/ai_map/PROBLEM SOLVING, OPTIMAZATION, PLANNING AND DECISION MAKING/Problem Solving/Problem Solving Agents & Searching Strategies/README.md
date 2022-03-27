# Problem Solving Agents & Searching Strategies

Suchstrategien welche von Agenten eingesetzt werden koennen, sind beispielhaft:
- Informierte/Uninformierte Suche
- Adverseriale Suche
- Constraint basierte Suche 

## Uninformierte Suche 

Die Klassische Suche, traversiert den gegebenen Graphen, ohne einen Indikator wo der Zielknoten sich befindet.
Es wird dementsprechend wie in einer der wohl bekanntesten Suchen, der Breiten- oder Tiefensuche, nach fester Suchstrategie der Graph travesiert, bis das Ziel erreicht wurde. 

## Informierte Suche
Im Gegensatz zu der Uninformierten Suche, nutzt die Informierte Suche heuristiken, welche versuchen den Loesungsraum einzugrenzen.
Demnach ist durchschnittlich die Anzahl an ueberprueften moeglichkeiten geringer als die in der Uninformierten Suche.

## Adverseriale Suche

Die Adverseriale Suche befasst sich mit der Suche in einem Kompetitiven umfeld. Also in einem solchen, in dem mindestens ein "Gegenspieler" seine eigenen Ziele verfolgt.

## Constraint basierte Suche

Constraint basiert Suche (haeufig formuliert als CSP), sucht die Parameterkonfiguration, in welcher alle dem Problem auferlegten Einschraenkungen erfuellt werden.