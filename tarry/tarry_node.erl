-module(tarry_node).
-export([main/2]).

%%
%% NODE PROCESS
%%

main(MainPid, NodeName) ->
    % Entrypoint for node
    % Wait to receive list of connected nodes
    receive
        {connections, ConnectedNodes} ->
            % Send the acknowledgement that the connected nodes have been received
            MainPid ! {ack},
            % Enter the main tarry node function once we know which nodes we are connected to
            tarry(NodeName, ConnectedNodes)
    end.

tarry(NodeName, ConnectedNodes) ->
    % Wait to receive the Token for the first time
    receive
        {tarry, Sender, ReceivedToken} ->
            % Add our Name to the Token
            NewToken = [NodeName|ReceivedToken],
            % Initialise queue of nodes to send the Token to as all connected nodes except the parent
            SendQueue = lists:keydelete(Sender, 2, ConnectedNodes),
            case SendQueue of
                [] ->
                    % Only connected to the sender, so we can just send the Token straight back
                    Sender ! {tarry, self(), NewToken};
                [{_, ChildPid}|SendQueueTl] ->
                    % Send the Token to our first child
                    ChildPid ! {tarry, self(), NewToken},
                    % Enter the tarry aux function to wait for the Token to return
                    tarry_aux(NodeName, Sender, SendQueueTl)
            end
    end.

tarry_aux(NodeName, Parent, []) ->
    % We have sent the Token to all children, so once we receive it again we are done
    % Wait to receive the Token
    receive
        {tarry, _, ReceivedToken} ->
            % Add our Name to the Token
            NewToken = [NodeName|ReceivedToken],
            % We have sent the Token to all children, so send it back to our Parent
            Parent ! {tarry, self(), NewToken}
    end;

tarry_aux(NodeName, Parent, [{_, ChildPid}|SendQueueTl]) ->
    % We still have at least one child that we can send the Token to
    % Wait to receive the Token
    receive
        {tarry, _, ReceivedToken} ->
            % Add our Name to the Token
            NewToken = [NodeName|ReceivedToken],
            % Forward the Token on to our next remaining child
            ChildPid ! {tarry, self(), NewToken},
            % Recurse to wait for the Token to be received again
            tarry_aux(NodeName, Parent, SendQueueTl)
    end.
