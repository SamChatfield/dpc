-module(tarry_node).
-export([main/2]).

%%
%% NODE PROCESS
%%

main(MainPid, NodeName) ->
    receive
        {connections, ConnectedNodes} ->
            MainPid ! {ack},
            tarry(NodeName, ConnectedNodes)
    end.

tarry(NodeName, ConnectedNodes) ->
    receive
        {tarry, Sender, ReceivedToken} ->
            NewToken = [NodeName|ReceivedToken],
            SendQueue = lists:keydelete(Sender, 2, ConnectedNodes),
            case SendQueue of
                [] ->
                    Sender ! {tarry, self(), NewToken};
                [{_, ChildPid}|SendQueueTl] ->
                    ChildPid ! {tarry, self(), NewToken},
                    tarry_aux(NodeName, Sender, SendQueueTl)
            end
    end.

tarry_aux(NodeName, Parent, []) ->
    receive
        {tarry, _, ReceivedToken} ->
            NewToken = [NodeName|ReceivedToken],
            Parent ! {tarry, self(), NewToken}
    end;

tarry_aux(NodeName, Parent, [{_, ChildPid}|SendQueueTl]) ->
    receive
        {tarry, _, ReceivedToken} ->
            NewToken = [NodeName|ReceivedToken],
            ChildPid ! {tarry, self(), NewToken},
            tarry_aux(NodeName, Parent, SendQueueTl)
    end.
